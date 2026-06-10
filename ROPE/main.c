#include <math.h>
#include <stdio.h>
#include "rope.h"
#include "redefine.h"

/* =============================================================================
 * DEVICE BUFFER DECLARATIONS
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
static mat_t manual_Q_out[N_Q_HEADS][HEAD_DIM];
static mat_t manual_K_out[N_KV_HEADS][HEAD_DIM];

#ifndef DEBUG_CODE
#define DEBUG_CODE 0
#endif

/* =============================================================================
 * MANUAL REFERENCE IMPLEMENTATION
 * =============================================================================*/
static void manual_rope(void) {
    /* Step 1 + 2: inv_freq, cos, sin. */
    float ref_inv_freq[HALF];
    float ref_cos[HALF];
    float ref_sin[HALF];

    for (int i = 0; i < HALF; i++) {
        float exponent   = (2.0f * (float)i) / (float)ROTARY_DIM;
        ref_inv_freq[i]  = 1.0f / powf(ROPE_THETA, exponent);
        float angle      = ref_inv_freq[i] * (float)position;
        ref_cos[i]       = cosf(angle);
        ref_sin[i]       = sinf(angle);
    }

    /* Step 3: rotate Q heads. */
    for (int h = 0; h < N_Q_HEADS; h++) {
        for (int i = 0; i < HALF; i++) {
            float c = ref_cos[i], s = ref_sin[i];
            float a = Q_in[h][i], b = Q_in[h][i + HALF];
            manual_Q_out[h][i]        = a * c - b * s;
            manual_Q_out[h][i + HALF] = b * c + a * s;
        }
        /* Pass-through beyond ROTARY_DIM. */
        for (int i = ROTARY_DIM; i < HEAD_DIM; i++) {
            manual_Q_out[h][i] = Q_in[h][i];
        }

#if DEBUG_CODE
        printf("manual_Q_out[%d][0] = %f\n", h, manual_Q_out[h][0]);
#endif
    }

    /* Step 4: rotate K heads. */
    for (int h = 0; h < N_KV_HEADS; h++) {
        for (int i = 0; i < HALF; i++) {
            float c = ref_cos[i], s = ref_sin[i];
            float a = K_in[h][i], b = K_in[h][i + HALF];
            manual_K_out[h][i]        = a * c - b * s;
            manual_K_out[h][i + HALF] = b * c + a * s;
        }
        /* Pass-through beyond ROTARY_DIM. */
        for (int i = ROTARY_DIM; i < HEAD_DIM; i++) {
            manual_K_out[h][i] = K_in[h][i];
        }

#if DEBUG_CODE
        printf("manual_K_out[%d][0] = %f\n", h, manual_K_out[h][0]);
#endif
    }
}

int main() {

    position = 4;

    printf("=== RoPE Kernel Test ===\n");
    printf("N_Q_HEADS=%d  N_KV_HEADS=%d  HEAD_DIM=%d\n",
           N_Q_HEADS, N_KV_HEADS, HEAD_DIM);
    printf("ROTARY_DIM=%d  HALF=%d  ROPE_THETA=%.0f  position=%d\n\n",
           ROTARY_DIM, HALF, (float)ROPE_THETA, position);

    for (int h = 0; h < N_Q_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            Q_in[h][d] = (float)((h + 1) * (d + 1)) / (float)HEAD_DIM;
#if DEBUG_CODE
            printf("Q_in[%d][%d] = %f\n", h, d, Q_in[h][d]);
#endif
        }
    }

    for (int h = 0; h < N_KV_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            if (h % 2 == 0) {
                K_in[h][d] = (d % 2 == 0) ? 1.0f : 0.0f;
            } else {
                K_in[h][d] = (float)(d + 1) / (float)HEAD_DIM;
            }
#if DEBUG_CODE
            printf("K_in[%d][%d] = %f\n", h, d, K_in[h][d]);
#endif
        }
    }

    /* -------------------------------------------------------------------------
     * Manual RoPE computation on host (reference for verification).
     * -------------------------------------------------------------------------*/
    manual_rope();

    /* -------------------------------------------------------------------------
     * Launch the HyperOps kernel on the Redefine IP chip.
     * -------------------------------------------------------------------------*/
    printf("--- Launching HyperOps kernel ---\n");
    redefine_initialize(1, 1);
    __re_StartHyperOpInit(0, 0, 1, 1);
    __REDEFINE_main();
    redefine_execute();
    printf("Kernel execution complete.\n\n");

    /* -------------------------------------------------------------------------
     * Validate Q_out against manual reference.
     * -------------------------------------------------------------------------*/
    int valid = 1;

    for (int h = 0; h < N_Q_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
#if DEBUG_CODE
            printf("Q_out[%d][%d] = %f  |  manual_Q_out[%d][%d] = %f\n",
                   h, d, Q_out[h][d], h, d, manual_Q_out[h][d]);
#endif
            if (Q_out[h][d] != manual_Q_out[h][d]) {
                valid = 0;
                printf("[MISMATCH] Q_out[%d][%d]: chip=%f  ref=%f\n",
                       h, d, Q_out[h][d], manual_Q_out[h][d]);
            }
        }
    }

    /* -------------------------------------------------------------------------
     * Validate K_out against manual reference.
     * -------------------------------------------------------------------------*/
    for (int h = 0; h < N_KV_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
#if DEBUG_CODE
            printf("K_out[%d][%d] = %f  |  manual_K_out[%d][%d] = %f\n",
                   h, d, K_out[h][d], h, d, manual_K_out[h][d]);
#endif
            if (K_out[h][d] != manual_K_out[h][d]) {
                valid = 0;
                printf("[MISMATCH] K_out[%d][%d]: chip=%f  ref=%f\n",
                       h, d, K_out[h][d], manual_K_out[h][d]);
            }
        }
    }

    /* -------------------------------------------------------------------------
     * Print final result.
     * -------------------------------------------------------------------------*/
    if (valid)
        printf("RoPE computation successful.\n");
    else
        printf("RoPE computation unsuccessful!\n");

    return 0;
}