*#ifndef ATT_WSUM_H
#define ATT_WSUM_H

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

#endif /* ATT_WSUM_H */