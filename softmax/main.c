#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common_defs.h"
#include "redefine.h"

/* KERNEL OPERANDS IN DEVICE */

extern mat_t score[BATCH_SIZE][HEAD_NUM][Q_LEN][K_LEN];
extern mat_t output[BATCH_SIZE][HEAD_NUM][Q_LEN][K_LEN];

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
  // Load scores.
  float* dbg_score =
      load_vector_f32("debug_dumps/dbg_pre_softmax_l1_p7.bin", K_LEN);
  if(dbg_score) {
    printf("Loaded debug scores successfully\n");
  } else {
    printf("Did not load debug scores successfully\n");
    free(dbg_score);
    return 1;
  }
  // Load outputs.
  float* dbg_output =
      load_vector_f32("debug_dumps/dbg_post_softmax_l1_p7.bin", K_LEN);
  if(dbg_output) {
    printf("Loaded debug outputs successfully\n");
  } else {
    printf("Did not load debug outputs successfully\n");
    free(dbg_score);
    free(dbg_output);
    return 1;
  }

  // Initialise input score matrix.
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int h = 0; h < HEAD_NUM; h++) {
      for(int q = 0; q < Q_LEN; q++) {
        for(int k = 0; k < K_LEN; k++) {
          score[b][h][q][k] = dbg_score[k];
#if DEBUG_CODE
          if(k % 100 == 0)
            printf("score[%d][%d][%d][%d] = %f\n", b, h, q, k,
                   score[b][h][q][k]);
#endif
        }
      }
    }
  }

  redefine_initialize(1, 1);          // Initialise REDEFINE.
  __re_StartHyperOpInit(0, 0, 1, 1);  // Initialise Start HyperOp.
  __REDEFINE_main();                  // Execute Start HyperOp.
  redefine_execute();                 // Execute HyperOps.

  // Check output validity.
  int valid = 0;
  float tol = 1e-6f;
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int h = 0; h < HEAD_NUM; h++) {
      for(int q = 0; q < Q_LEN; q++) {
        for(int k = 0; k < K_LEN; k++) {
          if(output[b][h][q][k]!=dbg_output[k]){
            printf("not exact match");
            float diff = fabsf(output[b][h][q][k] - dbg_output[k]);
            if(diff <= tol) {
            valid++;
          }else{
            printf("outputs dont match");
          }
        }
         else {
          valid++;
         }
        }
      }
    }
  }

  printf("Valid: %d\n", valid);
  if(valid == K_LEN * HEAD_NUM)
    printf("Softmax successful.\n");
  else
    printf("Softmax unsuccessful!\n");

  free(dbg_score);
  free(dbg_output);

  return 0;
}
