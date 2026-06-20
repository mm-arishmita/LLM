#ifndef ATT_H
#define ATT_H

/* Dimension of each Q / K vector (head dimension). */
#define HEAD_DIM 128

/* Number of attention heads. */
#define ATTN_HEADS 24

/* Number of tokens in the sequence. Scale this to increase token count. */
#define SEQ_LEN 16

#define CE_USE_NUM 1

/* Dimensions handled per CE. */
#define INC_SIZE (HEAD_DIM / CE_USE_NUM)

/* Total ReduceSum HyperOps = one per (head, token) pair. */
#define TOTAL_REDUCE_OPS (ATTN_HEADS * SEQ_LEN)

/* Floating point type used for all tensors. */
typedef float mat_t;

#endif /* ATT_H */
