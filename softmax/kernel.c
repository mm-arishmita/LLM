#include <math.h>

#include "common_defs.h"
#include "redefine.h"

/* KERNEL DEFINITIONS */

// Number of Compute Elements to be used for processing.
#define CE_USE_NUM 4
// Amount of parallelisation.
#define USAGE_SIZE 4
// Number of score vector elements to be processed by 1 CE.
#define INC_SIZE (K_LEN / USAGE_SIZE)
// Kernel Termination Status.
#define FAIL 1
#define PASS 0

/* HELPER FUNCTIONS AND DEFINITIONS */

// Whether to use custom exponentiation function or not.
// #define CUSTOM_EXP

#ifdef CUSTOM_EXP
float exp_coeff[11] = {1.0f,
                       1.0f,
                       0.5f,
                       0.16666667163372039794921875f,
                       0.0416666679084300994873046875f,
                       0.008333333767950534820556640625f,
                       0.001388888922519981861114501953125f,
                       0.000198412701138295233249664306640625f,
                       0.000024801587642286904156208038330078125f,
                       0.000002755731884462875314056873321533203125f,
                       0.00000027557319981497130356729030609130859375f};
#endif  // CUSTOM_EXP

/// @brief Calculate approximate \f$e^x\f$.
/// @param x Value of exponent.
/// @param limit Number of steps to expand exponential series.
/// @return Approximation of \f$e^x\f$.
float exp_lim(float x, int limit) {
  float result;

#ifdef CUSTOM_EXP
  if(x == INFINITY) {
    result = INFINITY;
  } else if(x == -INFINITY) {
    result = 0;
  } else {
    result = 1.0f;       // Accumulator base case of limit as 0.
    float value = 1.0f;  // Series element base case of limit as 0.

    for(int i = 1; i <= limit; i++) {
      value *= x;
      result += value * exp_coeff[i];
    }
  }
#else
  result = expf(x);
#endif  // CUSTOM_EXP

  return result;
}

#define EXP_LIM 10
#define EXP(x) exp_lim(x, EXP_LIM)

/* HYPEROP FUNCTION DECLARATIONS */

__hyperOp__ void End();
__hyperOp__ void FindMax(__Op32, __Op32, __Op32, __Op32);
__hyperOp__ void Exponent(__Op32, __Op32, __Op32, __Op32, __Op32, __Op32,
                          __Op32);
__hyperOp__ void Sum(__Op32, __Op32, __Op32, __Op32);
__hyperOp__ void Softmax(__Op32, __Op32, __Op32, __Op32, __Op32, __Op32,
                         __Op32);

/* SMD DECLARATIONS */

__SMD__ smd_End = {.arity = 1, .ann = ANN_JOIN, .fptr = (__HyOpFunc)End};
__SMD__ smd_FindMax = {
    .arity = 4, .ann = ANN_NONE, .fptr = (__HyOpFunc)FindMax};
__SMD__ smd_Exponent = {
    .arity = 7, .ann = ANN_NONE, .fptr = (__HyOpFunc)Exponent};
__SMD__ smd_Sum = {.arity = 5, .ann = ANN_JOIN, .fptr = (__HyOpFunc)Sum};
__SMD__ smd_Softmax = {
    .arity = 7, .ann = ANN_NONE, .fptr = (__HyOpFunc)Softmax};

/* SHARED BUFFERS COMING FROM HOST */

__attribute__((aligned(64))) mat_t score[BATCH_SIZE][HEAD_NUM][Q_LEN][K_LEN];
__attribute__((aligned(64))) mat_t output[BATCH_SIZE][HEAD_NUM][Q_LEN][K_LEN];

/* TEMPORARY BUFFERS */

__attribute__((aligned(64))) mat_t exp_vec[BATCH_SIZE][HEAD_NUM][Q_LEN][K_LEN];

/* HYPEROP FUNCTION DEFINITIONS */

/// Start HyperOp.
__kernel void __REDEFINE_main() {
#ifdef CUSTOM_EXP
  re_println("Custom Function Used.");
#endif

  // Create End HyperOp.
  __CMAddr EndFr = __createInst(&smd_End);

  // Set synchronisation wait count for End HyperOp Frame.
  __sync(re_opAddr(EndFr, 15), BATCH_SIZE * HEAD_NUM * Q_LEN * USAGE_SIZE);

  // Initialise hyperOps.
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int h = 0; h < HEAD_NUM; h++) {
      for(int q = 0; q < Q_LEN; q++) {
        // Create FindMax HyperOp.
        __CMAddr FindMaxFr = __createInst(&smd_FindMax);

        // Pass operands to the FindMax HyperOp Frames.
        __writeCM(re_opAddr(FindMaxFr, 0), b);                     // batch_id
        __writeCM(re_opAddr(FindMaxFr, 1), h);                     // head_id
        __writeCM(re_opAddr(FindMaxFr, 2), q);                     // query_id
        __writeCM(re_opAddr(FindMaxFr, 3), re_opAddr(EndFr, 15));  // endFr
      }
    }
  }
}

/// @brief Find max element per score vector
/// @param batch_id Batch index.
/// @param head_id Head index.
/// @param query_id Query index.
/// @param endFr Context Memory Event Frame of End hyperOp.
__hyperOp__ void FindMax(__Op32 batch_id, __Op32 head_id, __Op32 query_id,
                         __Op32 endFr) {
  // Indices.
  int b = (int)batch_id.i32;
  int h = (int)head_id.i32;
  int q = (int)query_id.i32;

  // Get maximum value in score state vector.
  float max_val = score[b][h][q][0];
  for(int i = 1; i < K_LEN; i++) {
    if(score[b][h][q][i] > max_val) {
      max_val = score[b][h][q][i];
    }
  }

  // Create Sum HyperOp.
  __CMAddr SumFr = __createInst(&smd_Sum);

  // Set synchronisation wait count for Sum HyperOp Frame.
  __sync(re_opAddr(SumFr, 15), USAGE_SIZE);

  // Pass operands to the Sum HyperOp Frame.
  __writeCM(re_opAddr(SumFr, 0), b);             // batch_id
  __writeCM(re_opAddr(SumFr, 1), h);             // head_id
  __writeCM(re_opAddr(SumFr, 2), q);             // query_id
  __writeCM(re_opAddr(SumFr, 3), endFr.cmAddr);  // endFr

  for(int i = 0; i < USAGE_SIZE; i++) {
    // Create Exponent HyperOp frame.
    __CMAddr ExponentFr = __fAllocWithCe(1, i % CE_USE_NUM);
    __fBind(ExponentFr, &smd_Exponent);

    // Pass operands to the Exponent HyperOp Frames.
    __writeCM(re_opAddr(ExponentFr, 0), b);                     // batch_id
    __writeCM(re_opAddr(ExponentFr, 1), h);                     // head_id
    __writeCM(re_opAddr(ExponentFr, 2), q);                     // query_id
    __writeCM(re_opAddr(ExponentFr, 3), i * INC_SIZE);          // base
    __writeCM(re_opAddr(ExponentFr, 4), (i + 1) * INC_SIZE);    // bound
    __writeCM(re_opAddr(ExponentFr, 5), max_val);               // max_val
    __writeCM(re_opAddr(ExponentFr, 6), re_opAddr(SumFr, 15));  // sumFr
  }
}

/// @brief Perform exponentiation.
/// @param batch_id Batch index.
/// @param head_id Head index.
/// @param query_id Query index.
/// @param base Base index.
/// @param bound Bound index.
/// @param max_val Value of maximum element.
/// @param sumFr Context Memory Event Frame of Sum hyperOp.
__hyperOp__ void Exponent(__Op32 batch_id, __Op32 head_id, __Op32 query_id,
                          __Op32 base, __Op32 bound, __Op32 max_val,
                          __Op32 sumFr) {
  // Indices.
  int batch = (int)batch_id.i32;
  int head = (int)head_id.i32;
  int query = (int)query_id.i32;

  // Exponentiate each element of the score state.
  for(int k = (int)base.i32; k < (int)bound.i32; k++) {
    exp_vec[batch][head][query][k] =
        EXP(score[batch][head][query][k] - max_val.f32);
  }

  // Signal current hyperOp completion to the Sum hyperOp.
  __sync(sumFr.cmAddr, -1);

  // Delete current HyperOp.
  __fDelete(re_getSelfID());
}

/// @brief Calculate sum of exponents.
/// @param batch_id Batch index.
/// @param head_id Head index.
/// @param query_id Query index.
/// @param endFr Context Memory Event Frame of End hyperOp.
__hyperOp__ void Sum(__Op32 batch_id, __Op32 head_id, __Op32 query_id,
                     __Op32 endFr) {
  // Indices.
  int batch = (int)batch_id.i32;
  int head = (int)head_id.i32;
  int query = (int)query_id.i32;

  // Sum all exponentiated elements.
  float exp_sum = 0.0f;
  for(int k = 0; k < K_LEN; k++) {
    exp_sum += exp_vec[batch][head][query][k];
  }

  // Calculate inverse of the sum.
  float inv_exp_sum = 1.0f / exp_sum;

  // Create Softmax HyperOps.
  for(int i = 0; i < USAGE_SIZE; i++) {
    __CMAddr SoftmaxFr = __fAllocWithCe(1, (i / INC_SIZE) % CE_USE_NUM);
    __fBind(SoftmaxFr, &smd_Softmax);

    __writeCM(re_opAddr(SoftmaxFr, 0), batch);               // batch_id
    __writeCM(re_opAddr(SoftmaxFr, 1), head);                // head_id
    __writeCM(re_opAddr(SoftmaxFr, 2), query);               // query_id
    __writeCM(re_opAddr(SoftmaxFr, 3), i * INC_SIZE);        // base
    __writeCM(re_opAddr(SoftmaxFr, 4), (i + 1) * INC_SIZE);  // bound
    __writeCM(re_opAddr(SoftmaxFr, 5), inv_exp_sum);         // exp_scale
    __writeCM(re_opAddr(SoftmaxFr, 6), endFr.cmAddr);        // endFr
  }
}

/// @brief Calculate Softmax.
/// @param batch_id Batch index.
/// @param head_id Head index.
/// @param query_id Query index.
/// @param base Base index.
/// @param bound Bound index.
/// @param exp_scale Scale factor.
/// @param endFr Context Memory Event Frame of End hyperOp.
__hyperOp__ void Softmax(__Op32 batch_id, __Op32 head_id, __Op32 query_id,
                         __Op32 base, __Op32 bound, __Op32 exp_scale,
                         __Op32 endFr) {
  // Indices.
  int batch = (int)batch_id.i32;
  int head = (int)head_id.i32;
  int query = (int)query_id.i32;

  // Perform Softmax on each element.
  for(int k = (int)base.i32; k < (int)bound.i32; k++) {
    output[batch][head][query][k] =
        exp_vec[batch][head][query][k] * exp_scale.f32;
  }

  // Decrement synchronisation wait count for End HyperOp.
  __sync(endFr.cmAddr, -1);

  // Delete current HyperOp.
  __fDelete(re_getSelfID());
}

/// End HyperOp.
__hyperOp__ void End() { re_sigEndOfKernel(PASS); }
