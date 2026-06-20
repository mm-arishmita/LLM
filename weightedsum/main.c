/* =============================================================================
 * main_wsum.c  —  Host Code for Weighted Sum Kernel
 * =============================================================================*/
#include <math.h>
#include <stdio.h>

#include "wsum.h"
#include "redefine.h"

/* =============================================================================
 * DEVICE BUFFER DECLARATIONS
 * =============================================================================*/

extern mat_t Scores[ATTN_HEADS][SEQ_LEN];                  /* Softmax weights (input)    */
extern mat_t V[ATTN_HEADS][SEQ_LEN][HEAD_DIM];             /* Value cache    (input)     */
extern mat_t partials_v[ATTN_HEADS][CE_USE_NUM][HEAD_DIM]; /* CE scratch                 */
extern mat_t Ctx[ATTN_HEADS][HEAD_DIM];                    /* Output: context vectors    */

/* =============================================================================
 * HOST-ONLY BUFFER
 * =============================================================================*/

mat_t manual_ctx[ATTN_HEADS][HEAD_DIM];

#ifndef DEBUG_CODE
#define DEBUG_CODE 0
#endif

/* =============================================================================
 * MANUAL IMPLEMENTATION
 *
 * Reference weighted sum on the host CPU.
 *   manual_ctx[h][d] = Σ_{tok=0}^{SEQ_LEN-1}  Scores[h][tok] * V[h][tok][d]
 * =============================================================================*/
static void manual_weighted_sum() {
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0;
            for (int tok = 0; tok < SEQ_LEN; tok++) {
                acc += (float)Scores[h][tok] * (float)V[h][tok][d];
            }
            manual_ctx[h][d] = (float)acc;

#if DEBUG_CODE
            printf("manual_ctx[%d][%d] = %f\n", h, d, manual_ctx[h][d]);
#endif
        }
    }
}

int main() {

    /* -------------------------------------------------------------------------
     * 1. Initialise Scores (softmax weights).
     *    Each head gets a distinct pattern; rows are manually normalised
     *    so that Σ_tok Scores[h][tok] == 1  (mirrors softmax output).
     *
     *    Pattern:  raw[h][tok] = (h + 1) * (tok + 1)
     *    Normalise: Scores[h][tok] = raw[h][tok] / Σ_tok raw[h][tok]
     *    (different scale per head → distinct weighted sums per head)
     * -------------------------------------------------------------------------*/
    for (int h = 0; h < ATTN_HEADS; h++) {
        /* Compute raw unnormalised weights. */
        double row_sum = 0.0;
        for (int tok = 0; tok < SEQ_LEN; tok++) {
            Scores[h][tok] = (float)((h + 1) * (tok + 1));
            row_sum += Scores[h][tok];
        }
        /* Normalise so the row sums to 1. */
        for (int tok = 0; tok < SEQ_LEN; tok++) {
            Scores[h][tok] = (float)((double)Scores[h][tok] / row_sum);
#if DEBUG_CODE
            printf("Scores[%d][%d] = %f\n", h, tok, Scores[h][tok]);
#endif
        }
    }

    /* -------------------------------------------------------------------------
     * 2. Initialise V (value) buffers.
     *    Each (token, dim) gets a distinct pattern so heads produce
     *    different context vectors.
     *
     *    V[h][tok][d] = (tok % 2 == 0) ? (d % 2 == 0 ? 1.0f : 0.0f)   (alternating)
     *                                   : (float)(d + 1) / HEAD_DIM      (ramp)
     *    (mirrors K initialisation pattern from att_main.c)
     * -------------------------------------------------------------------------*/
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int tok = 0; tok < SEQ_LEN; tok++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                if (tok % 2 == 0) {
                    /* Even tokens: alternating 1/0 pattern. */
                    V[h][tok][d] = (d % 2 == 0) ? 1.0f : 0.0f;
                } else {
                    /* Odd tokens: linear ramp. */
                    V[h][tok][d] = (float)(d + 1) / (float)HEAD_DIM;
                }
#if DEBUG_CODE
                printf("V[%d][%d][%d] = %f\n", h, tok, d, V[h][tok][d]);
#endif
            }
        }
    }

    /* -------------------------------------------------------------------------
     * 3. Manual weighted sum computation.
     * -------------------------------------------------------------------------*/
    manual_weighted_sum();

    /* -------------------------------------------------------------------------
     * 4. Launch the HyperOps kernel on the Redefine IP chip.
     * -------------------------------------------------------------------------*/
    redefine_initialize(2, 2);           /* redefine initialisation            */
    __re_StartHyperOpInit(0, 0, 2, 2);  /* initialise start hyperop           */
    __REDEFINE_main();                   /* execute start hyperop              */
    redefine_execute();                  /* execute hyperops                   */

    /* -------------------------------------------------------------------------
     * 5. Validate chip output against manual output.
     * -------------------------------------------------------------------------*/
    int valid = 1;

    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
#if DEBUG_CODE
            printf("Ctx[%d][%d] = %f  |  manual_ctx[%d][%d] = %f\n",
                   h, d, Ctx[h][d], h, d, manual_ctx[h][d]);
#endif
            if (Ctx[h][d] != manual_ctx[h][d]) {
                valid = 0;
                /* Print mismatch even without DEBUG_CODE so failures are visible. */
                printf("[MISMATCH] Ctx[%d][%d]: chip=%f  ref=%f\n",
                       h, d, Ctx[h][d], manual_ctx[h][d]);
            }
        }
    }

    /* -------------------------------------------------------------------------
     * 6. Print final result.
     * -------------------------------------------------------------------------*/
    if (valid)
        printf("Weighted Sum computation successful.\n");
    else
        printf("Weighted Sum computation unsuccessful!\n");

    return 0;
}
