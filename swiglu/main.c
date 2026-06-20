#include <math.h>
#include <stdio.h>

#include "common_defs.h"
#include "redefine.h"

/* KERNEL OPERANDS IN DEVICE */

extern mat_t gate[BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE];
extern mat_t up[BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE];
extern mat_t output[BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE];

/* HOST DEFINITIONS */

#ifndef DEBUG_CODE
#define DEBUG_CODE 0
#endif

/* HOST CODE */

float* load_vector_f32(const char* path, int expected_size) {
  FILE* f = fopen(path, "rb");
  if(!f) {
    perror("fopen failed");
    return NULL;
  }

  float* data = (float*)malloc(sizeof(float) * expected_size);
  if(!data) {
    fclose(f);
    return NULL;
  }

  size_t read = fread(data, sizeof(float), expected_size, f);
  fclose(f);

  if(read != (size_t)expected_size) {
    fprintf(stderr, "Size mismatch: expected %d, got %zu\n", expected_size,
            read);
    free(data);
    return NULL;
  }

  return data;
}

/// Main function.
int main() {
  // Initialise up matrix.
  float* dbg_up =
      load_vector_f32("debug_dumps/dbg_mlp_up_l1_p0.bin", HIDDEN_SIZE);
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int s = 0; s < SEQ_LEN; s++) {
      for(int i = 0; i < HIDDEN_SIZE; i++) {
        up[b][s][i] = dbg_up[i];
#if DEBUG_CODE
        if(i % 100 == 0) {
          printf("up[%d][%d][%d] = %f\n", b, s, i, up[b][s][i]);
        }
#endif
      }
    }
  }

  // Initialise gate matrix.
  float* dbg_gate = load_vector_f32("debug_dumps/dbg_mlp_gate_l1_p0.bin", HIDDEN_SIZE);
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int s = 0; s < SEQ_LEN; s++) {
      for(int i = 0; i < HIDDEN_SIZE; i++) {
        gate[b][s][i] = dbg_gate[i];
#if DEBUG_CODE
        if(i % 100 == 0) {
          printf("gate[%d][%d][%d] = %f\n", b, s, i, gate[b][s][i]);
        }
#endif
      }
    }
  }

  redefine_initialize(1, 1);          // Initialise REDEFINE.
  __re_StartHyperOpInit(0, 0, 1, 1);  // Initialise Start HyperOp.
  __REDEFINE_main();                  // Execute Start HyperOp.
  redefine_execute();                 // Execute HyperOps.

  // Check output validity.
  float* dbg_out =
      load_vector_f32("debug_dumps/dbg_mlp_out_l1_p0.bin", HIDDEN_SIZE);
  int valid = 0;
  float tol = 1e-6f;
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int s = 0; s < SEQ_LEN; s++) {
      for(int i = 0; i < HIDDEN_SIZE; i++) {
        if(output[b][s][i] != dbg_out[i]) {
            float diff = fabsf(output[b][s][i] - dbg_out[i]);
	    printf("not\n");
	    if(diff <= tol) {
		printf("results match with tolerance\n");
           	 valid++;
           }
#if DEBUG_CODE
          printf(
              "dbg_out[%d] = %+1.16f is not equal to "
              "output[%d][%d][%d] = %+1.16f\n",
              i, dbg_out[i], b, s, i, output[b][s][i]);
#endif
        } else {
          valid++;
        }
      }
    }
  }

  printf("Valid: %d\n", valid);
  if(valid == SEQ_LEN * HIDDEN_SIZE)
    printf("SwiGLU successful.\n");
  else
    printf("SwiGLU unsuccessful!\n");

  free(dbg_up);
  free(dbg_out);
  free(dbg_gate);

  return 0;
}
