#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "att.h"
#include "redefine.h"

/* =============================================================================
 * DEVICE BUFFER DECLARATIONS
 * =============================================================================
 * K layout changed from synthetic test:
 *   OLD: K[SEQ_LEN][HEAD_DIM]               — shared flat table
 *   NEW: K[ATTN_HEADS][SEQ_LEN][HEAD_DIM]   — per Q-head slice (GQA-expanded)
 *
 * The extractor writes K as float32[ATTN_HEADS][SEQ_LEN][HEAD_DIM] so each
 * Q head already has the correct KV-head row copied in — no GQA indexing
 * needed inside att.c.
 * =============================================================================*/

extern mat_t Q[ATTN_HEADS][HEAD_DIM];                    /* Query vectors              */
extern mat_t K[ATTN_HEADS][SEQ_LEN][HEAD_DIM];          /* Key vectors (GQA-expanded) */
extern mat_t partials[ATTN_HEADS][SEQ_LEN][CE_USE_NUM]; /* CE scratch                 */
extern mat_t Scores[ATTN_HEADS][SEQ_LEN];               /* Chip output scores         */

/* =============================================================================
 * HOST-ONLY BUFFERS
 * =============================================================================*/
static mat_t ref_scores[ATTN_HEADS][SEQ_LEN];   /* Golden scores from extractor  */
static mat_t manual_scores[ATTN_HEADS][SEQ_LEN]; /* Re-computed reference on host */

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
 * Re-computes attention scores from loaded Q and K on the host CPU.
 * Serves as a second sanity check alongside ref_scores from the extractor.
 * =============================================================================*/
static void manual_attention_scores(void) {
    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            double acc = 0.0;
            for (int d = 0; d < HEAD_DIM; d++) {
                acc += (double)Q[h][d] * (double)K[h][t][d];
            }
            manual_scores[h][t] = (float)(acc * (double)inv_sqrt);
        }
    }
}

/* =============================================================================
 * COMPARE HELPER
 * Compares two score arrays element-wise with a float tolerance.
 * Returns 1 if all match, 0 if any mismatch.
 * =============================================================================*/
static int compare_scores(const mat_t a[ATTN_HEADS][SEQ_LEN],
                          const mat_t b[ATTN_HEADS][SEQ_LEN],
                          float tol,
                          const char *label_a,
                          const char *label_b) {
    int ok = 1;
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            float diff = fabsf(a[h][t] - b[h][t]);
            if (diff > tol) {
                ok = 0;
                printf("[MISMATCH] h=%d t=%d : %s=%.8f  %s=%.8f  diff=%.2e\n",
                       h, t, label_a, a[h][t], label_b, b[h][t], diff);
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
     *    Override:  ./attn_test  dump_L1_pos2_Q.bin  dump_L1_pos2_K.bin  dump_L1_pos2_scores.bin
     * -------------------------------------------------------------------------*/
    const char *q_path      = (argc > 1) ? argv[1] : "dump_L1_pos0_Q.bin";
    const char *k_path      = (argc > 2) ? argv[2] : "dump_L1_pos0_K.bin";
    const char *scores_path = (argc > 3) ? argv[3] : "dump_L1_pos0_scores.bin";

    printf("=== Phi-4-mini Layer-1 Attention Score Test ===\n");
    printf("ATTN_HEADS=%d  HEAD_DIM=%d  SEQ_LEN=%d  CE_USE_NUM=%d\n\n",
           ATTN_HEADS, HEAD_DIM, SEQ_LEN, CE_USE_NUM);

    /* -------------------------------------------------------------------------
     * 1. Load Q  [ATTN_HEADS][HEAD_DIM]
     * -------------------------------------------------------------------------*/
    printf("Loading Q from '%s'...\n", q_path);
    if (load_f32_bin(q_path, (float *)Q,
                     (size_t)ATTN_HEADS * HEAD_DIM) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 2. Load K  [ATTN_HEADS][SEQ_LEN][HEAD_DIM]   (GQA-expanded by extractor)
     * -------------------------------------------------------------------------*/
    printf("Loading K from '%s'...\n", k_path);
    if (load_f32_bin(k_path, (float *)K,
                     (size_t)ATTN_HEADS * SEQ_LEN * HEAD_DIM) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 3. Load reference scores  [ATTN_HEADS][SEQ_LEN]
     * -------------------------------------------------------------------------*/
    printf("Loading reference scores from '%s'...\n", scores_path);
    if (load_f32_bin(scores_path, (float *)ref_scores,
                     (size_t)ATTN_HEADS * SEQ_LEN) != 0) return 1;

    printf("All inputs loaded successfully.\n\n");

    /* -------------------------------------------------------------------------
     * 4. Re-compute scores manually on host (second sanity check).
     * -------------------------------------------------------------------------*/
    manual_attention_scores();

    /* -------------------------------------------------------------------------
     * 5. Check manual vs extractor reference  (catches load/layout errors).
     * -------------------------------------------------------------------------*/
    printf("--- Check 1: manual_scores vs extractor ref_scores (tol=1e-4) ---\n");
    int check1 = compare_scores(manual_scores, ref_scores, 1e-4f,
                                "manual", "ref");
    if (check1)
        printf("  PASS: manual recompute matches extractor reference.\n\n");
    else
        printf("  FAIL: layout or load error — fix before running chip.\n\n");

#if DEBUG_CODE
    printf("--- Q (first head) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  Q[0][%d] = %f\n", d, Q[0][d]);
    printf("--- K (head 0, tok 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  K[0][0][%d] = %f\n", d, K[0][0][d]);
    printf("--- ref_scores ---\n");
    for (int h = 0; h < ATTN_HEADS; h++)
        for (int t = 0; t < SEQ_LEN; t++)
            printf("  ref_scores[%d][%d] = %f\n", h, t, ref_scores[h][t]);
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
     * 7. Check chip output vs extractor reference scores.
     * -------------------------------------------------------------------------*/
    printf("--- Check 2: chip Scores vs extractor ref_scores (tol=1e-4) ---\n");
    int check2 = compare_scores(Scores, ref_scores, 1e-4f,
                                "chip", "ref");

    /* -------------------------------------------------------------------------
     * 8. Also check chip vs manual (belt-and-suspenders).
     * -------------------------------------------------------------------------*/
    printf("--- Check 3: chip Scores vs manual_scores (tol=1e-4) ---\n");
    int check3 = compare_scores(Scores, manual_scores, 1e-4f,
                                "chip", "manual");

    /* -------------------------------------------------------------------------
     * 9. Final verdict.
     * -------------------------------------------------------------------------*/
    printf("\n=== Result ===\n");
    if (check1 && check2 && check3)
        printf("Attention Score computation SUCCESSFUL.\n");
    else {
        if (!check1) printf("  [FAIL] Check 1: manual vs ref — data load/layout issue.\n");
        if (!check2) printf("  [FAIL] Check 2: chip vs ref  — HyperOps computation error.\n");
        if (!check3) printf("  [FAIL] Check 3: chip vs manual — HyperOps computation error.\n");
        printf("Attention Score computation UNSUCCESSFUL.\n");
    }

    return (check1 && check2 && check3) ? 0 : 1;
}