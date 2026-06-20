/* =============================================================================
 * PARALLELISM STRATEGY:
 *   One ReduceSum HyperOp is created per (head, token) pair.
 *   Each ReduceSum spawns CE_USE_NUM MulOp HyperOps that each compute a
 *   partial dot product over INC_SIZE = HEAD_DIM / CE_USE_NUM dimensions.
 *   ReduceSum (ANN_JOIN) waits for all MulOps, sums partials, applies
 *   1/sqrt(HEAD_DIM) scaling, writes Scores[h][t], then signals End.
 *   End (ANN_JOIN) waits for all ATTN_HEADS * SEQ_LEN ReduceSum signals.
 * =============================================================================*/

#include <math.h>
#include "att.h"
#include "redefine.h"

#define PASS 0
#define FAIL 1

/* =============================================================================
 * HYPEROP FUNCTION DECLARATIONS
 * =============================================================================*/

__hyperOp__ void End();
__hyperOp__ void MulOp(__Op32, __Op32, __Op32, __Op32, __Op32, __Op32);
__hyperOp__ void ReduceSum(__Op32, __Op32, __Op32);

/* =============================================================================
 * SMD (Static Metadata) DECLARATIONS
 * =============================================================================*/
__SMD__ smd_End = {.arity = 1, .ann   = ANN_JOIN, .fptr  = (__HyOpFunc)End};
__SMD__ smd_MulOp = {.arity = 6, .ann   = ANN_NONE, .fptr  = (__HyOpFunc)MulOp};
__SMD__ smd_ReduceSum = {.arity = 4, .ann   = ANN_JOIN, .fptr  = (__HyOpFunc)ReduceSum};

/* =============================================================================
 * SHARED BUFFERS  
 * =============================================================================*/

__attribute__((aligned(64))) mat_t Q[ATTN_HEADS][HEAD_DIM];
__attribute__((aligned(64))) mat_t K[ATTN_HEADS][SEQ_LEN][HEAD_DIM];
__attribute__((aligned(64))) mat_t partials[ATTN_HEADS][SEQ_LEN][CE_USE_NUM];
__attribute__((aligned(64))) mat_t Scores[ATTN_HEADS][SEQ_LEN];

/* =============================================================================
 *  HYPEROPS FRAMES
 * =============================================================================*/

__CMAddr MulOpFrames[ATTN_HEADS][SEQ_LEN][CE_USE_NUM];
__CMAddr ReduceSumFrames[ATTN_HEADS][SEQ_LEN];

/* =============================================================================
 * START HYPEROP 
 *   1. Create End frame.
 *   2. Create all ReduceSum frames; set each sync count = CE_USE_NUM.
 *   3. Create all MulOp frames, pinned to CEs.
 *   4. Set End sync count = TOTAL_REDUCE_OPS.
 *   5. Write MulOp operands  → MulOps fire immediately (ANN_NONE).
 *   6. Write ReduceSum operands (already blocked by sync count from step 2).
 * =============================================================================*/
__kernel void __REDEFINE_main() {

    //ENDFRAME//
    __CMAddr EndFr = __createInst(&smd_End);

    // Counter for total number of ReduceSum HyperOps signalling End.
    int end_sync_count = 0;

    //Create one ReduceSum frame per (head, token) pair.
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            __CMAddr ReduceFr = __createInst(&smd_ReduceSum);
            ReduceSumFrames[h][t] = ReduceFr;

            //COUNTER FOR number of MULOP HYPEROPS
            int reduce_sync_count = 0;
        
           //Create one MulOp frame per (head, token, CE) triple.
           //Pin each frame to a specific CE for explicit data parallelism.
           //Round-robin: ce % CE_USE_NUM ensures even CE load distribution.
            for (int ce = 0; ce < CE_USE_NUM; ce++) {
                __CMAddr MulFr = __fAllocWithCe(1, ce % CE_USE_NUM);
                __fBind(MulFr, &smd_MulOp);
                MulOpFrames[h][t][ce] = MulFr;

                //increment counter to keep traxck of mul hyperop
                reduce_sync_count++;
            }
            //set synchronisation wait count for Reduce hyperop
             __sync(re_opAddr(ReduceFr, 15), reduce_sync_count);

        
             /* ------------------------------------------------------------------
            Write MulOp operands → all MulOps fire in parallel (ANN_NONE).
            Slot layout:
            0 — head_id     : attention head index
            1 — tok_id      : token index
            2 — ce_id       : CE index (identifies which partial slot to write)
            3 — base        : start index in HEAD_DIM for this CE's chunk
            4 — bound       : end index (exclusive) for this CE's chunk
            5 — reduceSumFr : CMAddr of parent ReduceSum's sync slot (slot 15)
           ------------------------------------------------------------------ */
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
            } // Each ReduceSum will signal End once — accumulate total.
            end_sync_count++;

        }
    }
    /* ------------------------------------------------------------------
     *  Set End sync count = total ReduceSum HyperOps (ATTN_HEADS * SEQ_LEN).
     *    MUST be set before writing ReduceSum operands in step 6.
     * ------------------------------------------------------------------ */
    __sync(re_opAddr(EndFr, 15), end_sync_count);

    
    /* ------------------------------------------------------------------
     *  Write ReduceSum operands.
     *    ReduceSum frames are blocked on their sync count (step 2) so
     *    writing operands here is safe — they will not fire until all
     *    CE_USE_NUM MulOps have decremented the sync count to zero.
     *
     *    Slot layout:
     *      0 — head_id : attention head index
     *      1 — tok_id  : token index
     *      2 — endFr   : CMAddr of End HyperOp's sync slot (slot 15)
     * ------------------------------------------------------------------ */
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            __writeCM(re_opAddr(ReduceSumFrames[h][t], 0), h);
            __writeCM(re_opAddr(ReduceSumFrames[h][t], 1), t);
            __writeCM(re_opAddr(ReduceSumFrames[h][t], 2),
                      re_opAddr(EndFr, 15));
        }
    }
}

/* =============================================================================
 * MULOP HYPEROP
 * @param head_id     Attention head index.
 * @param tok_id      Token index.
 * @param ce_id       CE index — identifies which partials slot to write.
 * @param base        Start index into HEAD_DIM for this CE's chunk.
 * @param bound       End index (exclusive) into HEAD_DIM for this CE's chunk.
 * @param reduceSumFr CMAddr of parent ReduceSum's sync slot (slot 15).
 * =============================================================================*/
__hyperOp__ void MulOp(__Op32 head_id,  __Op32 tok_id, __Op32 ce_id,
                        __Op32 base,     __Op32 bound,  __Op32 reduceSumFr) {
    int h  = (int)head_id.i32;
    int t  = (int)tok_id.i32;
    int ce = (int)ce_id.i32;
    int lo = (int)base.i32;
    int hi = (int)bound.i32;

    /* Partial dot product: Q[h][lo..hi) · K[t][lo..hi)
     * Double accumulator matches reference C implementation precision.     */
    float acc = 0.0;
    for (int d = lo; d < hi; d++) {
        acc += (float)Q[h][d] * (float)K[h][t][d];
    }

    /* Write partial result to scratch buffer. */
    partials[h][t][ce] = (float)acc;

    /* Signal parent ReduceSum: this CE's chunk is complete. */
    __sync(reduceSumFr.cmAddr, -1);
}

/* =============================================================================
 * REDUCESUM HYPEROP  (ANN_JOIN)
 * @param head_id  Attention head index.
 * @param tok_id   Token index.
 * @param endFr    CMAddr of End HyperOp's sync slot (slot 15).
 * =============================================================================*/
__hyperOp__ void ReduceSum(__Op32 head_id, __Op32 tok_id, __Op32 endFr) {
    int h = (int)head_id.i32;
    int t = (int)tok_id.i32;

    /* Cleanup: delete non-intrinsic MulOp frames for this (head, token). */
    for (int ce = 0; ce < CE_USE_NUM; ce++) {
        __fDelete(MulOpFrames[h][t][ce]);
    }

    /* Accumulate all CE partial dot products. */
    float dot = 0.0;
    for (int ce = 0; ce < CE_USE_NUM; ce++) {
        dot += (float)partials[h][t][ce];
    }

    /* Apply attention scale: score = dot(Q[h], K[t]) / sqrt(HEAD_DIM). */
    float inv_sqrt_dim = 1.0f / sqrtf((float)HEAD_DIM);
    Scores[h][t] = (float)(dot * (float)inv_sqrt_dim);

    /* Signal End: this (head, token) score is finalised. */
    __sync(endFr.cmAddr, -1);
}

/* =============================================================================
 * END HYPEROP  (ANN_JOIN)
 *
 * Fires after all ATTN_HEADS * SEQ_LEN ReduceSum HyperOps have signalled.
 * Scores[h][t] is valid for all h in [0, ATTN_HEADS), t in [0, SEQ_LEN).
 * Cleans up all ReduceSum frames and signals host that the kernel is done.
 * =============================================================================*/
__hyperOp__ void End() {

     re_println("End HyperOp.\n");
     re_sigEndOfKernel(PASS);
}   
