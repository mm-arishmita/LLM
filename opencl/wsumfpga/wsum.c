//#include <math.h>
//#include "wsum.h"
#include "redefine_opencl.h"
#ifndef __FUNCTION_SIM
#include <pocl/pocl_device.h>
extern volatile __kernel_exec_cmd __kernel_command
    __attribute__((section("POCL_KERNEL_EXE_CMD")));
#endif

#define __REDEFINE

#define PASS 0
#define FAIL 1

extern volatile ReclKernelContext __reclKernelContext;
__CMAddr _endKernelEventId;

/* Dimension of each V vector (head dimension). */
#define HEAD_DIM 128

/* Number of attention heads. */
#define ATTN_HEADS 24

/* Number of tokens in the sequence. Scale this to increase token count. */
#define SEQ_LEN 20

/* Number of Compute Engines used for parallelism. */
#define CE_USE_NUM 4

/* Dimensions handled per CE. */
#define INC_SIZE (HEAD_DIM / CE_USE_NUM)

/* Total ReduceCtx HyperOps = one per head. */
#define TOTAL_REDUCE_OPS (ATTN_HEADS)

/* Floating point type used for all tensors. */
typedef float mat_t;

/* =============================================================================
 * HYPEROP FUNCTION DECLARATIONS
 * =============================================================================*/

__hyperOp__ void End();
__hyperOp__ void WtSumOp(__Op32, __Op32, __Op32, __Op32, __Op32);
__hyperOp__ void ReduceCtx(__Op32, __Op32);

/* =============================================================================
 * SMD (Static Metadata) DECLARATIONS
 * =============================================================================*/
__SMD__ smd_End = {
    .arity = 1,
    .ann   = ANN_JOIN,
    .fptr  = (__HyOpFunc)End
};

__SMD__ smd_WtSumOp = {
    .arity = 5,
    .ann   = ANN_NONE,
    .fptr  = (__HyOpFunc)WtSumOp
};

__SMD__ smd_ReduceCtx = {
    .arity = 3,                          /* head_id, dim_base, endFr + sync slot */
    .ann   = ANN_JOIN,
    .fptr  = (__HyOpFunc)ReduceCtx
};

/* =============================================================================
 * STATIC BUFFER POINTERS
 * =============================================================================*/
static mat_t *g_Scores;         
static mat_t *g_V;          
static mat_t *g_partials;  
static mat_t *g_ctx;     

#define IDX_Scores(h,t)      ((h) * SEQ_LEN  + (t))
#define IDX_V(t,d)      ((t) * HEAD_DIM  + (d))
#define IDX_P(h,ce,d)   (((h) * CE_USE_NUM + (ce)) * HEAD_DIM + (d))
#define IDX_ctx(h,d)      ((h) *  HEAD_DIM + (d))

/* =============================================================================
 * HYPEROP FRAMES
 *
 *   WtSumOpFrames[h][ce]   — one WtSumOp frame per (head, CE)
 *                             each covers INC_SIZE dims: [ce*INC_SIZE, (ce+1)*INC_SIZE)
 *   ReduceCtxFrames[h]     — one ReduceCtx frame per head
 *                             fires after all CE_USE_NUM WtSumOps for that head finish
 * =============================================================================*/
__CMAddr WtSumOpFrames[ATTN_HEADS][CE_USE_NUM];
__CMAddr ReduceCtxFrames[ATTN_HEADS];

/* =============================================================================
 * WTSUMOP HYPEROP  (ANN_NONE — fires immediately when all slots written)
 * =============================================================================*/
__hyperOp__ void WtSumOp(__Op32 head_id,     __Op32 ce_id,
                          __Op32 base,         __Op32 bound,
                          __Op32 reduceCtxFr) {
    int h  = (int)head_id.i32;
    int ce = (int)ce_id.i32;
    int lo = (int)base.i32;
    int hi = (int)bound.i32;

    /* For each dim d in this CE's chunk, accumulate weighted sum over all tokens.
     * Double accumulator matches reference C implementation precision.          */
    for (int d = lo; d < hi; d++) {
        float acc = 0.0;
        for (int tok = 0; tok < SEQ_LEN; tok++) {
            acc += (float)g_Scores[IDX_Scores(h,tok)] * (float)g_V[IDX_V(tok,d)];
        }
        /* Write partial result to CE's own slice — no race condition.           */
        g_partials[IDX_P(h,ce,d)] = (float)acc;
    }

    /* Signal parent ReduceCtx: this CE's chunk is complete. */
    __sync(reduceCtxFr.cmAddr, -1);
}

/* =============================================================================
 * REDUCECTX HYPEROP  (ANN_JOIN — fires after all CE_USE_NUM WtSumOps signal)
 * =============================================================================*/
__hyperOp__ void ReduceCtx(__Op32 head_id, __Op32 endFr) {
    int h = (int)head_id.i32;

    /* Cleanup: delete WtSumOp frames for this head. */
    for (int ce = 0; ce < CE_USE_NUM; ce++) {
        __fDelete(WtSumOpFrames[h][ce]);
    }

    /* Assemble final context vector by summing CE partial results.*/
    for (int d = 0; d < HEAD_DIM; d++) {
        float acc = 0.0;
        for (int ce = 0; ce < CE_USE_NUM; ce++) {
            acc += (float)g_partials[IDX_P(h,ce,d)];
        }
        g_ctx[IDX_ctx(h,d)] = (float)acc;
    }

    /* Signal End: context vector for head h is finalised. */
    __sync(endFr.cmAddr, -1);
}

/* =============================================================================
 * END HYPEROP  (ANN_JOIN)
 * =============================================================================*/
__hyperOp__ void End() {
    re_println("End HyperOp — weighted sum complete.\n");
    CE_FLUSH_ALL();
    LLC_FLUSH_ALL();
  __sync(_endKernelEventId, -1, memory_order_relaxed);
}




/* =============================================================================
 * wsum_start - FPGA ENTRY POINT
 * =============================================================================*/
__kernel void wsum_start(mat_t *Scores,         
                        mat_t *V,         
                        mat_t *partials,  
                        mat_t *ctx) 
{
     /* Capture host buffer pointers for use by asynchronous HyperOps. */
    g_Scores        = Scores;
    g_V        = V;
    g_partials = partials;
    g_ctx   = ctx;

    /* ── End frame ─────────────────────────────────────────────────────────── */
    __CMAddr EndFr = __createInst(&smd_End);
    int end_sync_count = 0;

    /* ── Create all ReduceCtx and WtSumOp frames ───────────────────────────── */
    for (int h = 0; h < ATTN_HEADS; h++) {

        /* One ReduceCtx frame per head.
         * Blocked until CE_USE_NUM WtSumOps decrement its sync to zero.       */
        __CMAddr ReduceFr = __createInst(&smd_ReduceCtx);
        ReduceCtxFrames[h] = ReduceFr;

        int reduce_sync_count = 0;

        /* One WtSumOp frame per CE, pinned round-robin. */
        for (int ce = 0; ce < CE_USE_NUM; ce++) {
            __CMAddr WtFr = __fAllocWithCe(1, ce % CE_USE_NUM);
            __fBind(WtFr, &smd_WtSumOp);
            WtSumOpFrames[h][ce] = WtFr;
            reduce_sync_count++;
        }

        /* Set ReduceCtx sync count = number of WtSumOps it must wait for. */
        __sync(re_opAddr(ReduceFr, 15), reduce_sync_count);

        /* ── Write WtSumOp operands → all fire immediately (ANN_NONE) ────── */
        for (int ce = 0; ce < CE_USE_NUM; ce++) {
            int base  = ce * INC_SIZE;          /* INC_SIZE = HEAD_DIM / CE_USE_NUM */
            int bound = base + INC_SIZE;

            __writeCM(re_opAddr(WtSumOpFrames[h][ce], 0), h);
            __writeCM(re_opAddr(WtSumOpFrames[h][ce], 1), ce);
            __writeCM(re_opAddr(WtSumOpFrames[h][ce], 2), base);
            __writeCM(re_opAddr(WtSumOpFrames[h][ce], 3), bound);
            __writeCM(re_opAddr(WtSumOpFrames[h][ce], 4),
                      re_opAddr(ReduceFr, 15));
            //__writeCM(re_opAddr(WtSumOpFrames[h][ce], 5), 0);  /* unused slot */
        }

        /* Each ReduceCtx signals End once. */
        end_sync_count++;
    }

    /* ── Set End sync count before writing ReduceCtx operands ─────────────── */
    __sync(re_opAddr(EndFr, 15), end_sync_count);

    /* ── Write ReduceCtx operands ──────────────────────────────────────────── 
     * Safe to write now: ReduceCtx frames are blocked on their sync counts,
     * so they cannot fire until all their WtSumOps have finished.            */
    for (int h = 0; h < ATTN_HEADS; h++) {
        __writeCM(re_opAddr(ReduceCtxFrames[h], 0), h);
        __writeCM(re_opAddr(ReduceCtxFrames[h], 1),
                  re_opAddr(EndFr, 15));
    }
}

