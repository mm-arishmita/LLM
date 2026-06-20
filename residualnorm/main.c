#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "common_defs.h"
#include "redefine.h"

/* KERNEL OPERANDS IN DEVICE */

extern mat_t input[BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE];
extern mat_t hidden[BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE];
extern mat_t output[BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE];
extern mat_t weights[HIDDEN_SIZE];

/* HOST DEFINITIONS */

#ifndef DEBUG_CODE
#define DEBUG_CODE 0
#endif

/* HOST CODE */

float* load_vector_f32(const char *path, int expected_size) {
  FILE *f = fopen(path, "rb");
  if(!f) {
    perror("fopen failed");
    return NULL;
  }

  float *data = (float*)malloc(sizeof(float) * expected_size);
  if(!data) {
    fclose(f);
    return NULL;
  }

  size_t read = fread(data, sizeof(float), expected_size, f);
  fclose(f);

  if(read != (size_t)expected_size) {
    fprintf(stderr, "Size mismatch: expected %d, got %zu\n",
            expected_size, read);
    free(data);
    return NULL;
  }

  return data;
}

void build_fname(char *out, size_t sz, const char *dir, const char *tag,
                 int layer, int pos) {
  snprintf(out, sz, "%s/dbg_%s_l%d_p%d.bin", dir, tag, layer, pos);
}

LayerDebug load_layer_debug(const char *dir, int layer, int pos, int hidden)
{
    LayerDebug d = {0};
    char fname[256];

    // ---- RMS 1 ----
    build_fname(fname, sizeof(fname), dir, "rms1", layer, pos);
    d.rms1_in = load_vector_f32(fname, hidden);

    build_fname(fname, sizeof(fname), dir, "rms1_out", layer, pos);
    d.rms1_out = load_vector_f32(fname, hidden);

    // ---- Residual 1 ----
    build_fname(fname, sizeof(fname), dir, "res1", layer, pos);
    d.res1_in = load_vector_f32(fname, hidden);

    build_fname(fname, sizeof(fname), dir, "res1_out", layer, pos);
    d.res1_out = load_vector_f32(fname, hidden);

    // ---- Attention output ----
    build_fname(fname, sizeof(fname), dir, "attn_out", layer, pos);
    d.attn_out = load_vector_f32(fname, hidden);

    // ---- RMS 2 ----
    build_fname(fname, sizeof(fname), dir, "rms2", layer, pos);
    d.rms2_in = load_vector_f32(fname, hidden);

    build_fname(fname, sizeof(fname), dir, "rms2_out", layer, pos);
    d.rms2_out = load_vector_f32(fname, hidden);

    // ---- Residual 2 ----
    build_fname(fname, sizeof(fname), dir, "res2", layer, pos);
    d.res2_in = load_vector_f32(fname, hidden);

    build_fname(fname, sizeof(fname), dir, "res2_out", layer, pos);
    d.res2_out = load_vector_f32(fname, hidden);

    // ---- MLP output ----
    build_fname(fname, sizeof(fname), dir, "mlp_out", layer, pos);
    d.mlp_out = load_vector_f32(fname, hidden);

    return d;
}

void free_layer_debug(LayerDebug *d) {
  free(d->rms1_in);
  free(d->rms1_out);
  free(d->res1_in);
  free(d->res1_out);
  free(d->rms2_in);
  free(d->rms2_out);
  free(d->res2_in);
  free(d->res2_out);
  free(d->attn_out);
  free(d->mlp_out);
}

/// Main function.
int main() {
  LayerDebug dbg = load_layer_debug("debug_dumps", 1, 0, HIDDEN_SIZE);
  if (dbg.rms1_in && dbg.rms1_out) {
    printf("Loaded debug tensors successfully\n");
  }
  else {
    printf("Did not load debug tensors successfully\n");
    free_layer_debug(&dbg);
    return 1;
  }

  // Initialise input and hidden matrix.
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int s = 0; s < SEQ_LEN; s++) {
      for(int i = 0; i < HIDDEN_SIZE; i++) {
        input[b][s][i] = dbg.rms1_in[i];
#if DEBUG_CODE
        printf("input[%d][%d][%d] = %f\n", b, s, i, input[b][s][i]);
#endif
        hidden[b][s][i] = dbg.attn_out[i];
#if DEBUG_CODE
        printf("hidden[%d][%d][%d] = %f\n", b, s, i, hidden[b][s][i]);
#endif
      }
    }
  }

  // Load weights.
  float *rms_weights = load_vector_f32("debug_dumps/rmsnorm_layer1.bin", HIDDEN_SIZE * 2);
  if (rms_weights) {
    printf("Loaded rms weights successfully\n");
  }
  else {
    printf("Did not load rms weights successfully\n");
    free_layer_debug(&dbg);
    free(rms_weights);
    return 1;
  }

  for(int i = 0; i < HIDDEN_SIZE; i++) {
    // Add HIDDEN_SIZE to read post_attention_layernorm weights.
    weights[i] = rms_weights[HIDDEN_SIZE + i];
  }

  redefine_initialize(1, 1);          // Initialise REDEFINE.
  __re_StartHyperOpInit(0, 0, 1, 1);  // Initialise Start HyperOp.
  __REDEFINE_main();                  // Execute Start HyperOp.
  redefine_execute();                 // Execute HyperOps.

  // Check output validity.
  int valid = 0;
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int s = 0; s < SEQ_LEN; s++) {
      for(int i = 0; i < HIDDEN_SIZE; i++) {
        if(output[b][s][i] != dbg.rms2_out[i]) {
#if DEBUG_CODE
          printf("dbg.rms2_out[%d][%d][%d] = %+2.5f is not equal to output[%d][%d][%d] = %+2.5f\n", b, s, i, dbg.rms2_out[i], b, s, i, output[b][s][i]);
#endif
        }
        else {
          valid++;
        }
      }
    }
  }

  printf("Valid: %d\n", valid);
  if(valid == BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE)
    printf("RMSNorm successful.\n");
  else
    printf("RMSNorm unsuccessful!\n");

  free_layer_debug(&dbg);
  free(rms_weights);

  return 0;
}
