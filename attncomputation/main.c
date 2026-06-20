
/* =============================================================================
 * att_main.c  —  Host Code for Attention Score Kernel
 * =============================================================================*/
#include <math.h>
#include <stdio.h>

#include "att.h"
#include "redefine.h"

/* =============================================================================
 * DEVICE BUFFER DECLARATIONS
 * =============================================================================*/

extern mat_t Q[ATTN_HEADS][HEAD_DIM];           /* Query vectors             */
extern mat_t K[SEQ_LEN][HEAD_DIM];              /* Key vectors               */
extern mat_t partials[ATTN_HEADS][SEQ_LEN][CE_USE_NUM]; /* CE scratch        */
extern mat_t Scores[ATTN_HEADS][SEQ_LEN];       /* Output: attention scores  */

/* =============================================================================
 * HOST-ONLY BUFFER
 * =============================================================================*/

mat_t manual_scores[ATTN_HEADS][SEQ_LEN];

#ifndef DEBUG_CODE
#define DEBUG_CODE 0
#endif

/* =============================================================================
 * MANUAL IMPLEMENTATION
 * =============================================================================*/
static void manual_attention_scores() {
    float inv_sqrt_dim = 1.0f / sqrtf((float)HEAD_DIM);

    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            double acc = 0.0;
            for (int d = 0; d < HEAD_DIM; d++) {
                acc += (double)Q[h][d] * (double)K[t][d];
            }
            manual_scores[h][t] = (float)(acc * (double)inv_sqrt_dim);

#if DEBUG_CODE
            printf("manual_attention_scores[%d][%d] = %f\n", h, t,
                   manual_scores[h][t]);
#endif
        }
    }
}

int main() {

    /* -------------------------------------------------------------------------
     * 1. Initialise Q (query) buffers.
     *    Each head gets a distinct pattern so heads produce different scores.
     *    Q[h][d] = (h + 1) * (d + 1) / HEAD_DIM
     *    (matches the spirit of the reference: Q rows = {1,2,3,4}, {5,6,7,8}…)
     * -------------------------------------------------------------------------*/
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            Q[h][d] = (float)((h + 1) * (d + 1)) / (float)HEAD_DIM;
#if DEBUG_CODE
            printf("Q[%d][%d] = %f\n", h, d, Q[h][d]);
#endif
        }
    }

    /* -------------------------------------------------------------------------
     * 2. Initialise K (key) buffers.
     *    Each token gets a distinct pattern to produce varied scores.
     *    K[t][d] = (t % 2 == 0) ? (d % 2 == 0 ? 1.0f : 0.0f)   (alternating)
     *                            : (float)(d + 1) / HEAD_DIM      (ramp)
     *    This mimics the variety in the reference K matrix.
     * -------------------------------------------------------------------------*/
    for (int t = 0; t < SEQ_LEN; t++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            if (t % 2 == 0) {
                /* Even tokens: alternating 1/0 pattern. */
                K[t][d] = (d % 2 == 0) ? 1.0f : 0.0f;
            } else {
                /* Odd tokens: linear ramp. */
                K[t][d] = (float)(d + 1) / (float)HEAD_DIM;
            }
#if DEBUG_CODE
            printf("K[%d][%d] = %f\n", t, d, K[t][d]);
#endif
        }
    }

    /* -------------------------------------------------------------------------
     * Manual Attention Computation
     * -------------------------------------------------------------------------*/
    manual_attention_scores();

    /* -------------------------------------------------------------------------
     * 4. Launch the HyperOps kernel on the Redefine IP chip.
     * -------------------------------------------------------------------------*/
    redefine_initialize(1, 1); //redefine initialisation
    __re_StartHyperOpInit(0, 0, 1, 1); //initialise start hyperop
    __REDEFINE_main(); //execute start hyperop
    redefine_execute(); //execute hyperops

    /* -------------------------------------------------------------------------
     * 5. Validate chip output against manual output.
     * -------------------------------------------------------------------------*/
    int valid = 1;

    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
#if DEBUG_CODE
            printf("Scores[%d][%d] = %f  |  manual_attn_scores[%d][%d] = %f\n",
                   h, t, Scores[h][t], h, t, manual_scores[h][t]);
#endif
            if (Scores[h][t] != manual_scores[h][t]) {
                valid = 0;
                /* Print mismatch even without DEBUG_CODE so failures are visible. */
                printf("[MISMATCH] Scores[%d][%d]: chip=%f  ref=%f\n",
                       h, t, Scores[h][t], manual_scores[h][t]);
            }
        }
    }

    /* -------------------------------------------------------------------------
     * 6. Print final result.
     * -------------------------------------------------------------------------*/
    if (valid)
        printf("Attention Score computation successful.\n");
    else
        printf("Attention Score computation unsuccessful!\n");

    return 0;
}
