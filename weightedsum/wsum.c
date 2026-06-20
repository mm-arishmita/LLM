
#include <math.h>
#include "wsum.h"
#include "redefine.h"

#define PASS 0
#define FAIL 1

/* =============================================================================
 * DIMENSION REFERENCE
 *
 *   Scores : [ATTN_HEADS][SEQ_LEN]            softmax weights (input, pre-filled)
 *   V      : [ATTN_HEADS][SEQ_LEN][HEAD_DIM]  value cache    (input, pre-filled)
 *
 *   Weighted sum per head h, per dim d:
 *       Ctx[h][d] = Σ_{tok=0}^{SEQ_LEN-1}  Scores[h][tok] * V[h][tok][d]
 *
 *   Output:
 *       Ctx : [ATTN_HEADS][HEAD_DIM]
 *
 *   Parallelism — split HEAD_DIM across CEs (mirrors score computation):
 *       WtSumOp[h][d_chunk][ce]  owns columns [base, bound) of HEAD_DIM,
 *       loops over all SEQ_LEN tokens, writes partial sums to partials_v[h][ce][d].
 *       ReduceCtx[h][d] sums CE partials → one element of Ctx[h].
 *
 *   INC_SIZE = HEAD_DIM / CE_USE_NUM   (chunk of dims per CE, same as score code)
 * =============================================================================*/

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
 * SHARED BUFFERS
 *
 *   INPUTS  (written by caller / softmax stage before this kernel fires):
 *       Scores[h][tok]        — softmax-normalised attention weights
 *       V[h][tok][d]          — value cache
 *
 *   SCRATCH (written by WtSumOp, read by ReduceCtx):
 *       partials_v[h][ce][d]  — partial weighted sum for CE chunk of dims
 *                               layout [ce][d] keeps each CE's writes contiguous
 *
 *   OUTPUT (written by ReduceCtx):
 *       Ctx[h][d]             — final context vector, shape [ATTN_HEADS][HEAD_DIM]
 * =============================================================================*/
__attribute__((aligned(64))) mat_t Scores[ATTN_HEADS][SEQ_LEN];
__attribute__((aligned(64))) mat_t V[ATTN_HEADS][SEQ_LEN][HEAD_DIM];
__attribute__((aligned(64))) mat_t partials_v[ATTN_HEADS][CE_USE_NUM][HEAD_DIM];
__attribute__((aligned(64))) mat_t Ctx[ATTN_HEADS][HEAD_DIM];

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
 * KERNEL MAIN
 *
 * Launch strategy (mirrors attention score kernel exactly):
 *
 *   For each head h:
 *       1. Create one ReduceCtx frame; sync count = CE_USE_NUM.
 *       2. Create CE_USE_NUM WtSumOp frames, pinned round-robin to CEs.
 *       3. Write WtSumOp operands → all CE_USE_NUM WtSumOps fire in parallel.
 *       4. ReduceCtx fires only after all CE_USE_NUM WtSumOps decrement its sync.
 *       5. ReduceCtx signals End when done.
 *
 *   End fires after all ATTN_HEADS ReduceCtx HyperOps have signalled.
 *   At that point Ctx[h][d] is valid for all h, d.
 *
 *   WtSumOp slot layout:
 *       0 — head_id     : attention head index h
 *       1 — ce_id       : CE index (identifies which partials_v slice to write)
 *       2 — base        : start index into HEAD_DIM for this CE's chunk
 *       3 — bound       : end index (exclusive) into HEAD_DIM for this CE's chunk
 *       4 — reduceCtxFr : CMAddr of parent ReduceCtx's sync slot (slot 15)
 *       5 — (unused)    : kept for arity = 6 symmetry with MulOp
 *
 *   ReduceCtx slot layout:
 *       0 — head_id : attention head index h
 *       1 — endFr   : CMAddr of End HyperOp's sync slot (slot 15)
 * =============================================================================*/
__kernel void __REDEFINE_main() {

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

/* =============================================================================
 * WTSUMOP HYPEROP  (ANN_NONE — fires immediately when all slots written)
 *
 * @param head_id     Attention head index h.
 * @param ce_id       CE index — selects which partials_v[h][ce] slice to write.
 * @param base        First dim index in this CE's chunk  [base, bound).
 * @param bound       One-past-last dim index in this CE's chunk.
 * @param reduceCtxFr CMAddr of parent ReduceCtx's sync slot (slot 15).
 * @param _unused     Padding slot for arity symmetry with original MulOp.
 *
 * Inner loop (for each dim d in [base, bound)):
 *       partial[d] = Σ_{tok=0}^{SEQ_LEN-1}  Scores[h][tok] * V[h][tok][d]
 *
 * Written to: partials_v[h][ce][d]   (contiguous in d for cache efficiency)
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
            acc += (float)Scores[h][tok] * (float)V[h][tok][d];
        }
        /* Write partial result to CE's own slice — no race condition.           */
        partials_v[h][ce][d] = (float)acc;
    }

    /* Signal parent ReduceCtx: this CE's chunk is complete. */
    __sync(reduceCtxFr.cmAddr, -1);
}

/* =============================================================================
 * REDUCECTX HYPEROP  (ANN_JOIN — fires after all CE_USE_NUM WtSumOps signal)
 *
 * @param head_id  Attention head index h.
 * @param endFr    CMAddr of End HyperOp's sync slot (slot 15).
 *
 * For each dim d in [0, HEAD_DIM):
 *       Ctx[h][d] = Σ_{ce=0}^{CE_USE_NUM-1}  partials_v[h][ce][d]
 *
 * Each CE only wrote to partials_v[h][ce][base..bound), so only those d values
 * are non-zero — summing across all CEs correctly assembles the full HEAD_DIM
 * context vector for head h.
 * =============================================================================*/
__hyperOp__ void ReduceCtx(__Op32 head_id, __Op32 endFr) {
    int h = (int)head_id.i32;

    /* Cleanup: delete WtSumOp frames for this head. */
    for (int ce = 0; ce < CE_USE_NUM; ce++) {
        __fDelete(WtSumOpFrames[h][ce]);
    }

    /* Assemble final context vector by summing CE partial results.
     * partials_v[h][ce][d] is non-zero only where CE ce was responsible for d,
     * so the sum across CEs gives the correct full Ctx[h][d].                 */
    for (int d = 0; d < HEAD_DIM; d++) {
        float acc = 0.0;
        for (int ce = 0; ce < CE_USE_NUM; ce++) {
            acc += (float)partials_v[h][ce][d];
        }
        Ctx[h][d] = (float)acc;
    }

    /* Signal End: context vector for head h is finalised. */
    __sync(endFr.cmAddr, -1);
}

/* =============================================================================
 * END HYPEROP  (ANN_JOIN)
 *
 * Fires after all ATTN_HEADS ReduceCtx HyperOps have signalled.
 * Ctx[h][d] is valid for all h in [0, ATTN_HEADS), d in [0, HEAD_DIM).
 *
 * Next step: concatenate Ctx[h] across heads → output projection W_O.
 *   Ctx reshaped to [ATTN_HEADS * HEAD_DIM] = [MODEL_DIM]
 * =============================================================================*/
__hyperOp__ void End() {
    re_println("End HyperOp — weighted sum complete.\n");
    re_sigEndOfKernel(PASS);
}
