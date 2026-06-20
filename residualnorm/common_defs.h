/* COMMON DEFINITIONS */

#define RMS_EPSILON 1e-5f
#define HIDDEN_SIZE 3072

#define BATCH_SIZE 1
#define SEQ_LEN 100

typedef float mat_t;

typedef struct {
    float *rms1_in;
    float *rms1_out;
    float *res1_in;
    float *res1_out;
    float *rms2_in;
    float *rms2_out;
    float *res2_in;
    float *res2_out;
    float *attn_out;
    float *mlp_out;
} LayerDebug;
