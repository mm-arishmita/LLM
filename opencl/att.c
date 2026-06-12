
/* ── FPGA fabric grid 
#define __NUMCOL__  2
#define __NUMROW__  2

#include <math.h>
#include "att.h"
#include "redefine.h"

#define PASS 0
#define FAIL 1

/* =============================================================================
 * HYPEROP DECLARATIONS 
 * =============================================================================*/
__hyperOp__ void End();
__hyperOp__ void MulOp(__Op32, __Op32, __Op32, __Op32, __Op32, __Op32);
__hyperOp__ void ReduceSum(__Op32, __Op32, __Op32);

/* =============================================================================
 * SMD DECLARATIONS
 * =============================================================================*/
__SMD__ smd_End       = { .arity = 1, .ann = ANN_JOIN, .fptr = (__HyOpFunc)End       };
__SMD__ smd_MulOp     = { .arity = 6, .ann = ANN_NONE, .fptr = (__HyOpFunc)MulOp     };
__SMD__ smd_ReduceSum = { .arity = 4, .ann = ANN_JOIN, .fptr = (__HyOpFunc)ReduceSum };

/* =============================================================================
 * FRAME ADDRESS TABLES 
 * =============================================================================*/
__CMAddr MulOpFrames    [ATTN_HEADS][SEQ_LEN][CE_USE_NUM];
__CMAddr ReduceSumFrames[ATTN_HEADS][SEQ_LEN];

/* =============================================================================
 * STATIC BUFFER POINTERS
 *
 * FSim att.c used global arrays:
 *   __attribute__((aligned(64))) mat_t Q[ATTN_HEADS][HEAD_DIM];
 *   __attribute__((aligned(64))) mat_t K[SEQ_LEN][HEAD_DIM];
 *   __attribute__((aligned(64))) mat_t partials[ATTN_HEADS][SEQ_LEN][CE_USE_NUM];
 *   __attribute__((aligned(64))) mat_t Scores[ATTN_HEADS][SEQ_LEN];
 *
 * FPGA att_fpga.c receives these as __kernel args and stores them here
 * so that MulOp / ReduceSum HyperOps can access them after att_start returns.
 * This is necessary because HyperOps are dispatched asynchronously by the
 * scheduler — they cannot receive the pointers as direct arguments.
 *
 * Flat-index macros match the row-major layout the host allocates:
 *   Q        [h * HEAD_DIM  + d]
 *   K        [t * HEAD_DIM  + d]
 *   partials [(h * SEQ_LEN + t) * CE_USE_NUM + ce]
 *   Scores   [h * SEQ_LEN  + t]
 * =============================================================================*/
static mat_t *g_Q;         
static mat_t *g_K;          
static mat_t *g_partials;  
static mat_t *g_Scores;     

#define IDX_Q(h,d)      ((h) * HEAD_DIM  + (d))
#define IDX_K(t,d)      ((t) * HEAD_DIM  + (d))
#define IDX_P(h,t,ce)   (((h) * SEQ_LEN + (t)) * CE_USE_NUM + (ce))
#define IDX_S(h,t)      ((h) * SEQ_LEN  + (t))

/* =============================================================================
 * MULOP HYPEROP 
 * =============================================================================*/
__hyperOp__ void MulOp(__Op32 head_id,  __Op32 tok_id,  __Op32 ce_id,
                        __Op32 base,     __Op32 bound,   __Op32 reduceSumFr)
{
    int h  = (int)head_id.i32;
    int t  = (int)tok_id.i32;
    int ce = (int)ce_id.i32;
    int lo = (int)base.i32;
    int hi = (int)bound.i32;

    /* Partial dot product: Q[h][lo..hi) · K[t][lo..hi)  — unchanged */
    double acc = 0.0;
    for (int d = lo; d < hi; d++) {
        acc += (double)g_Q[IDX_Q(h,d)] * (double)g_K[IDX_K(t,d)];
    }

    /* Write partial — unchanged: partials[h][t][ce] = (float)acc */
    g_partials[IDX_P(h,t,ce)] = (float)acc;

    /* Signal parent ReduceSum — unchanged */
    __sync(reduceSumFr.cmAddr, -1);
}

/* =============================================================================
 * REDUCESUM HYPEROP
 * =============================================================================*/
__hyperOp__ void ReduceSum(__Op32 head_id, __Op32 tok_id, __Op32 endFr)
{
    int h = (int)head_id.i32;
    int t = (int)tok_id.i32;

    /* Cleanup MulOp frames — unchanged */
    for (int ce = 0; ce < CE_USE_NUM; ce++) {
        __fDelete(MulOpFrames[h][t][ce]);
    }

    /* Accumulate all CE partials — unchanged */
    double dot = 0.0;
    for (int ce = 0; ce < CE_USE_NUM; ce++) {
        dot += (double)g_partials[IDX_P(h,t,ce)];
    }

    /* Scale and write final score — unchanged */
    float inv_sqrt_dim = 1.0f / sqrtf((float)HEAD_DIM);
    g_Scores[IDX_S(h,t)] = (float)(dot * (double)inv_sqrt_dim);

    /* Signal End — unchanged */
    __sync(endFr.cmAddr, -1);
}

/* =============================================================================
 * END HYPEROP 
 * =============================================================================*/
__hyperOp__ void End()
{
    re_println("End HyperOp.\n");
    re_sigEndOfKernel(PASS);
}

/* =============================================================================
 * att_start  —  FPGA ENTRY POINT
 * =============================================================================*/
__kernel void att_start(mat_t *Q,         
                        mat_t *K,         
                        mat_t *partials,  
                        mat_t *Scores)    
{
    /* Capture host buffer pointers for use by asynchronous HyperOps. */
    g_Q        = Q;
    g_K        = K;
    g_partials = partials;
    g_Scores   = Scores;

    
    re_println("start hyperop\n");

    /* Create End frame */
    __CMAddr EndFr = __createInst(&smd_End);
    int end_sync_count = 0;

    /* Create one ReduceSum frame per (head, token) pair */
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {

            __CMAddr ReduceFr = __createInst(&smd_ReduceSum);
            ReduceSumFrames[h][t] = ReduceFr;

            int reduce_sync_count = 0;

            /* Create one MulOp frame per CE, pinned round-robin */
            for (int ce = 0; ce < CE_USE_NUM; ce++) {
                __CMAddr MulFr = __fAllocWithCe(1, ce % CE_USE_NUM);
                __fBind(MulFr, &smd_MulOp);
                MulOpFrames[h][t][ce] = MulFr;
                reduce_sync_count++;
            }

            /* Set ReduceSum sync count */
            __sync(re_opAddr(ReduceFr, 15), reduce_sync_count);

            /* Write MulOp operands — fires immediately (ANN_NONE) */
            for (int ce = 0; ce < CE_USE_NUM; ce++) {
                int base  = ce * INC_SIZE;
                int bound = base + INC_SIZE;
                __writeCM(re_opAddr(MulOpFrames[h][t][ce], 0), h);
                __writeCM(re_opAddr(MulOpFrames[h][t][ce], 1), t);
                __writeCM(re_opAddr(MulOpFrames[h][t][ce], 2), ce);
                __writeCM(re_opAddr(MulOpFrames[h][t][ce], 3), base);
                __writeCM(re_opAddr(MulOpFrames[h][t][ce], 4), bound);
                __writeCM(re_opAddr(MulOpFrames[h][t][ce], 5),
                          re_opAddr(ReduceFr, 15));
            }

            end_sync_count++;
        }
    }

    /* Set End sync count BEFORE writing ReduceSum operands */
    __sync(re_opAddr(EndFr, 15), end_sync_count);

    /* Write ReduceSum operands */
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            __writeCM(re_opAddr(ReduceSumFrames[h][t], 0), h);
            __writeCM(re_opAddr(ReduceSumFrames[h][t], 1), t);
            __writeCM(re_opAddr(ReduceSumFrames[h][t], 2),
                      re_opAddr(EndFr, 15));
        }
    }
}
