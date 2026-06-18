#ifndef ROPE_H
#define ROPE_H

/* Number of Q attention heads. */
#define N_Q_HEADS   24

/* Number of KV attention heads. */
#define N_KV_HEADS  8

/* Dimension of each head vector. */
#define HEAD_DIM    128

/* Rotary embedding dimension (partial RoPE — only first ROTARY_DIM dims rotated). */
#define ROTARY_DIM  96

/* Half of rotary dim — number of dimension pairs rotated. */
#define HALF        (ROTARY_DIM / 2)

/* RoPE base frequency (fixed for Phi-4-mini). */
#define ROPE_THETA  10000.0f

/* Floating point type used for all tensors. */
typedef float mat_t;

#endif /* ROPE_H */