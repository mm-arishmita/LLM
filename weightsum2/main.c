#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wsum.h"
#include "redefine.h"

/* =============================================================================
 * DEVICE BUFFER DECLARATIONS
 * =============================================================================
 * Scores layout:
 *   Scores[ATTN_HEADS][SEQ_LEN]          — softmax-normalised attention weights
 *                                           output of softmax stage, input here
 *                                           each row sums to 1.0
 *
 * V layout:
 *   V[ATTN_HEADS][SEQ_LEN][HEAD_DIM]     — value vectors (GQA-expanded)
 *                                           extractor writes per Q-head slice,
 *                                           no GQA indexing needed inside wsum.c
 *
 * Output:
 *   Ctx[ATTN_HEADS][HEAD_DIM]            — context vectors, one per head
 *                                           Ctx[h][d] = Σ_tok Scores[h][tok]*V[h][tok][d]
 * =============================================================================*/

extern mat_t Scores[ATTN_HEADS][SEQ_LEN];                  /* Softmax weights (input)    */
extern mat_t V[ATTN_HEADS][SEQ_LEN][HEAD_DIM];             /* Value cache (GQA-expanded) */
extern mat_t partials_v[ATTN_HEADS][CE_USE_NUM][HEAD_DIM]; /* CE scratch                 */
extern mat_t Ctx[ATTN_HEADS][HEAD_DIM];                    /* Chip output context vecs   */

/* =============================================================================
 * HOST-ONLY BUFFERS
 * =============================================================================*/
static mat_t ref_ctx[ATTN_HEADS][HEAD_DIM];     /* Golden ctx from extractor     */
static mat_t manual_ctx[ATTN_HEADS][HEAD_DIM];  /* Re-computed reference on host */

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
 * Re-computes weighted sum from loaded Scores and V on the host CPU.
 * Serves as a second sanity check alongside ref_ctx from the extractor.
 *
 *   manual_ctx[h][d] = Σ_{tok=0}^{SEQ_LEN-1}  Scores[h][tok] * V[h][tok][d]
 * =============================================================================*/
static void manual_weighted_sum(void) {
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            double acc = 0.0;
            for (int tok = 0; tok < SEQ_LEN; tok++) {
                acc += (double)Scores[h][tok] * (double)V[h][tok][d];
            }
            manual_ctx[h][d] = (float)acc;
        }
    }
}

/* =============================================================================
 * COMPARE HELPER
 * Compares two Ctx arrays element-wise with a float tolerance.
 * Returns 1 if all match, 0 if any mismatch.
 * =============================================================================*/
static int compare_ctx(const mat_t a[ATTN_HEADS][HEAD_DIM],
                       const mat_t b[ATTN_HEADS][HEAD_DIM],
                       float tol,
                       const char *label_a,
                       const char *label_b) {
    int ok = 1;
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float diff = fabsf(a[h][d] - b[h][d]);
            if (diff > tol) {
                ok = 0;
                printf("[MISMATCH] h=%d d=%d : %s=%.8f  %s=%.8f  diff=%.2e\n",
                       h, d, label_a, a[h][d], label_b, b[h][d], diff);
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
     *    Override:
     *      ./wsum_test dump_L1_pos2_scores.bin dump_L1_pos2_V.bin dump_L1_pos2_ctx.bin
     *
     *    scores : softmax output from score stage  [ATTN_HEADS][SEQ_LEN]
     *    V      : value cache (GQA-expanded)       [ATTN_HEADS][SEQ_LEN][HEAD_DIM]
     *    ctx    : golden context from extractor    [ATTN_HEADS][HEAD_DIM]
     * -------------------------------------------------------------------------*/
    const char *scores_path = (argc > 1) ? argv[1] : "dump_L1_pos0_scores.bin";
    const char *v_path      = (argc > 2) ? argv[2] : "dump_L1_pos0_V.bin";
    const char *ctx_path    = (argc > 3) ? argv[3] : "dump_L1_pos0_ctx.bin";

    printf("=== Phi-4-mini Layer-1 Weighted Sum Test ===\n");
    printf("ATTN_HEADS=%d  HEAD_DIM=%d  SEQ_LEN=%d  CE_USE_NUM=%d\n\n",
           ATTN_HEADS, HEAD_DIM, SEQ_LEN, CE_USE_NUM);

    /* -------------------------------------------------------------------------
     * 1. Load Scores  [ATTN_HEADS][SEQ_LEN]
     *    Softmax-normalised weights — each row sums to 1.
     *    Produced by the attention score kernel (existing att kernel output).
     * -------------------------------------------------------------------------*/
    printf("Loading Scores from '%s'...\n", scores_path);
    if (load_f32_bin(scores_path, (float *)Scores,
                     (size_t)ATTN_HEADS * SEQ_LEN) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 2. Load V  [ATTN_HEADS][SEQ_LEN][HEAD_DIM]  (GQA-expanded by extractor)
     * -------------------------------------------------------------------------*/
    printf("Loading V from '%s'...\n", v_path);
    if (load_f32_bin(v_path, (float *)V,
                     (size_t)ATTN_HEADS * SEQ_LEN * HEAD_DIM) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 3. Load reference context vectors  [ATTN_HEADS][HEAD_DIM]
     * -------------------------------------------------------------------------*/
    printf("Loading reference ctx from '%s'...\n", ctx_path);
    if (load_f32_bin(ctx_path, (float *)ref_ctx,
                     (size_t)ATTN_HEADS * HEAD_DIM) != 0) return 1;

    printf("All inputs loaded successfully.\n\n");

    /* -------------------------------------------------------------------------
     * 4. Re-compute weighted sum manually on host (second sanity check).
     * -------------------------------------------------------------------------*/
    manual_weighted_sum();

    /* -------------------------------------------------------------------------
     * 5. Check manual vs extractor reference  (catches load/layout errors).
     * -------------------------------------------------------------------------*/
    printf("--- Check 1: manual_ctx vs extractor ref_ctx (tol=1e-4) ---\n");
    int check1 = compare_ctx(manual_ctx, ref_ctx, 1e-4f,
                             "manual", "ref");
    if (check1)
        printf("  PASS: manual recompute matches extractor reference.\n\n");
    else
        printf("  FAIL: layout or load error — fix before running chip.\n\n");

#if DEBUG_CODE
    printf("--- Scores (head 0) ---\n");
    for (int tok = 0; tok < SEQ_LEN; tok++)
        printf("  Scores[0][%d] = %f\n", tok, Scores[0][tok]);
    printf("--- V (head 0, tok 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  V[0][0][%d] = %f\n", d, V[0][0][d]);
    printf("--- ref_ctx (head 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  ref_ctx[0][%d] = %f\n", d, ref_ctx[0][d]);
#endif

    /* -------------------------------------------------------------------------
     * 6. Launch the HyperOps kernel on the Redefine IP chip.
     * -------------------------------------------------------------------------*/
    printf("--- Launching HyperOps kernel ---\n");
    redefine_initialize(1, 1);
    __re_StartHyperOpInit(0, 0, 1, 1);
    __REDEFINE_main();
    redefine_execute();
    printf("Kernel execution complete.\n\n");

    /* -------------------------------------------------------------------------
     * 7. Check chip output vs extractor reference ctx.
     * -------------------------------------------------------------------------*/
    printf("--- Check 2: chip Ctx vs extractor ref_ctx (tol=1e-4) ---\n");
    int check2 = compare_ctx(Ctx, ref_ctx, 1e-5f,
                             "chip", "ref");

    /* -------------------------------------------------------------------------
     * 8. Also check chip vs manual (belt-and-suspenders).
     * -------------------------------------------------------------------------*/
    printf("--- Check 3: chip Ctx vs manual_ctx (tol=1e-4) ---\n");
    int check3 = compare_ctx(Ctx, manual_ctx, 1e-5f,
                             "chip", "manual");

    /* -------------------------------------------------------------------------
     * 9. Final verdict.
     * -------------------------------------------------------------------------*/
    printf("\n=== Result ===\n");
    if (check1 && check2 && check3)
        printf("Weighted Sum computation SUCCESSFUL.\n");
    else {
        if (!check1) printf("  [FAIL] Check 1: manual vs ref   — data load/layout issue.\n");
        if (!check2) printf("  [FAIL] Check 2: chip vs ref     — HyperOps computation error.\n");
        if (!check3) printf("  [FAIL] Check 3: chip vs manual  — HyperOps computation error.\n");
        printf("Weighted Sum computation UNSUCCESSFUL.\n");
    }

    return (check1 && check2 && check3) ? 0 : 1;
}
