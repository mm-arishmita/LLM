#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rope.h"
#include "redefine.h"

/* =============================================================================
 * DEVICE BUFFER DECLARATIONS
 * =============================================================================
 * Q_in  / K_in  : raw Q and K vectors BEFORE RoPE, loaded from extractor dumps
 *                 Q_in  [N_Q_HEADS][HEAD_DIM]   — 24 heads × 128 dims
 *                 K_in  [N_KV_HEADS][HEAD_DIM]  —  8 heads × 128 dims (not GQA-expanded)
 *
 * Q_out / K_out : rotated output produced by the HyperOps kernel
 *
 * inv_freq / cos_cache / sin_cache : internal scratch computed by the kernel
 *                                    (NOT loaded from file — kernel owns these)
 *
 * position : token position used during inference — read from rope_meta.txt
 *            and passed to the kernel as an integer
 * =============================================================================*/
extern mat_t Q_in[N_Q_HEADS][HEAD_DIM];
extern mat_t K_in[N_KV_HEADS][HEAD_DIM];
extern mat_t Q_out[N_Q_HEADS][HEAD_DIM];
extern mat_t K_out[N_KV_HEADS][HEAD_DIM];
extern mat_t inv_freq[HALF];
extern mat_t cos_cache[HALF];
extern mat_t sin_cache[HALF];
extern int   position;

/* =============================================================================
 * HOST-ONLY BUFFERS
 * =============================================================================*/
static mat_t ref_Q_out[N_Q_HEADS][HEAD_DIM];   /* Golden Q_out from extractor   */
static mat_t ref_K_out[N_KV_HEADS][HEAD_DIM];  /* Golden K_out from extractor   */
static mat_t manual_Q_out[N_Q_HEADS][HEAD_DIM]; /* Re-computed reference on host */
static mat_t manual_K_out[N_KV_HEADS][HEAD_DIM];

#ifndef DEBUG_CODE
#define DEBUG_CODE 0
#endif

/* =============================================================================
 * BINARY LOADER
 * Reads exactly `count` float32 values from `path` into `buf`.
 * Returns 0 on success, -1 on failure.
 * =============================================================================*/
static int load_f32_bin(const char *path, float *buf, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open '%s'\n", path);
        return -1;
    }
    size_t got = fread(buf, sizeof(float), count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "[ERROR] '%s': expected %zu floats, got %zu\n",
                path, count, got);
        return -1;
    }
    return 0;
}

/* =============================================================================
 * MANUAL REFERENCE IMPLEMENTATION
 * Mirrors apply_rope_single() in model.c and manual_rope() in the original
 * rope test exactly:
 *
 *   inv_freq[i] = 1 / (ROPE_THETA ^ (2i / ROTARY_DIM))
 *   angle       = inv_freq[i] * position
 *   cos_c[i]    = cos(angle),   sin_c[i] = sin(angle)
 *
 *   Q_out[h][i]        = Q_in[h][i]      * cos_c[i] - Q_in[h][i+HALF] * sin_c[i]
 *   Q_out[h][i+HALF]   = Q_in[h][i+HALF] * cos_c[i] + Q_in[h][i]      * sin_c[i]
 *   (elements beyond ROTARY_DIM passed through unchanged)
 *
 * Same rotation applied to K heads.
 * =============================================================================*/
static void manual_rope(void) {
    float ref_inv_freq[HALF];
    float ref_cos[HALF];
    float ref_sin[HALF];

    /* Step 1: build inv_freq, cos, sin */
    for (int i = 0; i < HALF; i++) {
        float exponent   = (2.0f * (float)i) / (float)ROTARY_DIM;
        ref_inv_freq[i]  = 1.0f / powf(ROPE_THETA, exponent);
        float angle      = ref_inv_freq[i] * (float)position;
        ref_cos[i]       = cosf(angle);
        ref_sin[i]       = sinf(angle);
    }

    /* Step 2: rotate Q heads */
    for (int h = 0; h < N_Q_HEADS; h++) {
        for (int i = 0; i < HALF; i++) {
            float c = ref_cos[i], s = ref_sin[i];
            float a = Q_in[h][i], b = Q_in[h][i + HALF];
            manual_Q_out[h][i]        = a * c - b * s;
            manual_Q_out[h][i + HALF] = b * c + a * s;
        }
        /* Pass-through beyond ROTARY_DIM */
        for (int i = ROTARY_DIM; i < HEAD_DIM; i++)
            manual_Q_out[h][i] = Q_in[h][i];

#if DEBUG_CODE
        printf("manual_Q_out[%d][0] = %f\n", h, manual_Q_out[h][0]);
#endif
    }

    /* Step 3: rotate K heads */
    for (int h = 0; h < N_KV_HEADS; h++) {
        for (int i = 0; i < HALF; i++) {
            float c = ref_cos[i], s = ref_sin[i];
            float a = K_in[h][i], b = K_in[h][i + HALF];
            manual_K_out[h][i]        = a * c - b * s;
            manual_K_out[h][i + HALF] = b * c + a * s;
        }
        /* Pass-through beyond ROTARY_DIM */
        for (int i = ROTARY_DIM; i < HEAD_DIM; i++)
            manual_K_out[h][i] = K_in[h][i];

#if DEBUG_CODE
        printf("manual_K_out[%d][0] = %f\n", h, manual_K_out[h][0]);
#endif
    }
}

/* =============================================================================
 * COMPARE HELPER
 * Element-wise comparison with float tolerance.
 * Returns 1 if all match, 0 if any mismatch.
 * =============================================================================*/
static int compare_Q(const mat_t a[N_Q_HEADS][HEAD_DIM],
                     const mat_t b[N_Q_HEADS][HEAD_DIM],
                     float tol,
                     const char *label_a,
                     const char *label_b) {
    int ok = 1;
    for (int h = 0; h < N_Q_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float diff = fabsf(a[h][d] - b[h][d]);
            if (diff > tol) {
                ok = 0;
                //printf("[MISMATCH] Q h=%d d=%d : %s=%.8f  %s=%.8f  diff=%.2e\n",
                      // h, d, label_a, a[h][d], label_b, b[h][d], diff);
            }
        }
    }
    return ok;
}

static int compare_K(const mat_t a[N_KV_HEADS][HEAD_DIM],
                     const mat_t b[N_KV_HEADS][HEAD_DIM],
                     float tol,
                     const char *label_a,
                     const char *label_b) {
    int ok = 1;
    for (int h = 0; h < N_KV_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float diff = fabsf(a[h][d] - b[h][d]);
            if (diff > tol) {
                ok = 0;
                //printf("[MISMATCH] K h=%d d=%d : %s=%.8f  %s=%.8f  diff=%.2e\n",
                       //h, d, label_a, a[h][d], label_b, b[h][d], diff);
            }
        }
    }
    return ok;
}

/* =============================================================================
 * MAIN
 * =============================================================================*/
int main(int argc, char *argv[]) {

    /* -------------------------------------------------------------------------
     * 0. Parse optional dump-file paths from command line.
     *    Defaults match the extractor's naming convention for layer 1, pos 0.
     *    Override example (layer 1, position 4):
     *      ./rope_test dump_L1_pos4_Q_in.bin  dump_L1_pos4_K_in.bin \
     *                  dump_L1_pos4_Q_out.bin dump_L1_pos4_K_out.bin 4
     *
     *    argv[1] : Q_in.bin   float32 [N_Q_HEADS][HEAD_DIM]
     *    argv[2] : K_in.bin   float32 [N_KV_HEADS][HEAD_DIM]
     *    argv[3] : Q_out.bin  float32 [N_Q_HEADS][HEAD_DIM]   (golden reference)
     *    argv[4] : K_out.bin  float32 [N_KV_HEADS][HEAD_DIM]  (golden reference)
     *    argv[5] : position   integer (token position used during extraction)
     * -------------------------------------------------------------------------*/
    const char *q_in_path  = (argc > 1) ? argv[1] : "dump_L1_pos0_Q_in.bin";
    const char *k_in_path  = (argc > 2) ? argv[2] : "dump_L1_pos0_K_in.bin";
    const char *q_out_path = (argc > 3) ? argv[3] : "dump_L1_pos0_Q_out.bin";
    const char *k_out_path = (argc > 4) ? argv[4] : "dump_L1_pos0_K_out.bin";

    /* position: prefer command-line arg, else fall back to compile-time default */
#ifdef POSITION
    position = POSITION;
#else
    position = 0;
#endif
    if (argc > 5) position = atoi(argv[5]);

    printf("=== Phi-4-mini Layer-%d RoPE Kernel Test ===\n", 1);
    printf("N_Q_HEADS=%d  N_KV_HEADS=%d  HEAD_DIM=%d\n",
           N_Q_HEADS, N_KV_HEADS, HEAD_DIM);
    printf("ROTARY_DIM=%d  HALF=%d  ROPE_THETA=%.0f  position=%d\n\n",
           ROTARY_DIM, HALF, (float)ROPE_THETA, position);

    /* -------------------------------------------------------------------------
     * 1. Load Q_in  [N_Q_HEADS][HEAD_DIM]
     *    Q vectors BEFORE RoPE, straight out of QKV projection.
     * -------------------------------------------------------------------------*/
    printf("Loading Q_in  from '%s'...\n", q_in_path);
    if (load_f32_bin(q_in_path, (float *)Q_in,
                     (size_t)N_Q_HEADS * HEAD_DIM) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 2. Load K_in  [N_KV_HEADS][HEAD_DIM]
     *    K vectors BEFORE RoPE. Shape is [N_KV_HEADS][HEAD_DIM] = [8][128].
     *    NOT GQA-expanded — matches what the kernel receives.
     * -------------------------------------------------------------------------*/
    printf("Loading K_in  from '%s'...\n", k_in_path);
    if (load_f32_bin(k_in_path, (float *)K_in,
                     (size_t)N_KV_HEADS * HEAD_DIM) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 3. Load golden Q_out  [N_Q_HEADS][HEAD_DIM]
     *    Q after RoPE as computed by the Python extractor (reference).
     * -------------------------------------------------------------------------*/
    printf("Loading ref Q_out from '%s'...\n", q_out_path);
    if (load_f32_bin(q_out_path, (float *)ref_Q_out,
                     (size_t)N_Q_HEADS * HEAD_DIM) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 4. Load golden K_out  [N_KV_HEADS][HEAD_DIM]
     *    K after RoPE as computed by the Python extractor (reference).
     * -------------------------------------------------------------------------*/
    printf("Loading ref K_out from '%s'...\n", k_out_path);
    if (load_f32_bin(k_out_path, (float *)ref_K_out,
                     (size_t)N_KV_HEADS * HEAD_DIM) != 0) return 1;

    printf("All inputs loaded successfully.\n\n");

    /* -------------------------------------------------------------------------
     * 5. Re-compute RoPE manually on host (second sanity check).
     *    Uses the same formula as apply_rope_single() in model.c.
     * -------------------------------------------------------------------------*/
    manual_rope();

    /* -------------------------------------------------------------------------
     * 6. Check 1: manual host recompute vs extractor reference.
     *    Catches any load/layout mismatch or formula divergence.
     * -------------------------------------------------------------------------*/
    printf("--- Check 1: manual_out vs extractor ref_out (tol=1e-4) ---\n");
    int check1_Q = compare_Q(manual_Q_out, ref_Q_out, 1e-6f, "manual_Q", "ref_Q");
    int check1_K = compare_K(manual_K_out, ref_K_out, 1e-6f, "manual_K", "ref_K");
    int check1   = check1_Q && check1_K;
    if (check1)
        printf("  PASS: manual recompute matches extractor reference.\n\n");
    else
        printf("  FAIL: layout or formula mismatch — fix before running chip.\n\n");

#if DEBUG_CODE
    printf("--- Q_in (head 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  Q_in[0][%d] = %f\n", d, Q_in[0][d]);
    printf("--- K_in (head 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  K_in[0][%d] = %f\n", d, K_in[0][d]);
    printf("--- ref_Q_out (head 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  ref_Q_out[0][%d] = %f\n", d, ref_Q_out[0][d]);
    printf("--- ref_K_out (head 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  ref_K_out[0][%d] = %f\n", d, ref_K_out[0][d]);
#endif

    /* -------------------------------------------------------------------------
     * 7. Launch the HyperOps kernel on the Redefine IP chip.
     * -------------------------------------------------------------------------*/
    printf("--- Launching HyperOps kernel ---\n");
    redefine_initialize(1, 1);
    __re_StartHyperOpInit(0, 0, 1, 1);
    __REDEFINE_main();
    redefine_execute();
    printf("Kernel execution complete.\n\n");

    /* -------------------------------------------------------------------------
     * 8. Check 2: chip Q_out / K_out vs extractor reference.
     * -------------------------------------------------------------------------*/
    printf("--- Check 2: chip Q_out/K_out vs extractor ref (tol=1e-4) ---\n");
    int check2_Q = compare_Q(Q_out, ref_Q_out, 1e-4f, "chip_Q", "ref_Q");
    int check2_K = compare_K(K_out, ref_K_out, 1e-4f, "chip_K", "ref_K");
    int check2   = check2_Q && check2_K;

    /* -------------------------------------------------------------------------
     * 9. Check 3: chip Q_out / K_out vs manual host recompute.
     * -------------------------------------------------------------------------*/
    printf("--- Check 3: chip Q_out/K_out vs manual recompute (tol=1e-4) ---\n");
    int check3_Q = compare_Q(Q_out, manual_Q_out, 1e-4f, "chip_Q", "manual_Q");
    int check3_K = compare_K(K_out, manual_K_out, 1e-4f, "chip_K", "manual_K");
    int check3   = check3_Q && check3_K;

    /* -------------------------------------------------------------------------
     * 10. Final verdict.
     * -------------------------------------------------------------------------*/
    printf("\n=== Result ===\n");
    if (check1 && check2 && check3) {
        printf("RoPE computation SUCCESSFUL.\n");
    } else {
        if (!check1) printf("  [FAIL] Check 1: manual vs ref    — data load/layout issue.\n");
        if (!check2) printf("  [FAIL] Check 2: chip vs ref      — HyperOps computation error.\n");
        if (!check3) printf("  [FAIL] Check 3: chip vs manual   — HyperOps computation error.\n");
        printf("RoPE computation UNSUCCESSFUL.\n");
    }

    return (check1 && check2 && check3) ? 0 : 1;
}
