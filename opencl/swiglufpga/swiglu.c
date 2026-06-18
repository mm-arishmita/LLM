#include "redefine_opencl.h"
#ifndef __FUNCTION_SIM
#include <pocl/pocl_device.h>
extern volatile __kernel_exec_cmd __kernel_command
    __attribute__((section("POCL_KERNEL_EXE_CMD")));
#endif

#define __REDEFINE

#define PASS 0
#define FAIL 1

extern volatile ReclKernelContext __reclKernelContext;
__CMAddr _endKernelEventId;

/* =============================================================================
 * KERNEL-LOCAL CONSTANTS
 * (Redefined here rather than relying on common_defs.h resolving at OpenCL
 *  build time — same approach as wsum.c/rms.c/softmax.c. Must match the
 *  values in common_defs.h used by the host.)
 * =============================================================================*/

#define HIDDEN_SIZE 8192
#define BATCH_SIZE 1
#define SEQ_LEN 1

/* Floating point type used for all tensors. */
typedef float mat_t;

/* Number of Compute Elements to be used for processing. */
#define CE_USE_NUM 4
/* Amount of parallelisation. */
#define USAGE_SIZE 8
/* Number of score vector elements to be processed by 1 CE. */
#define INC_SIZE (HIDDEN_SIZE / USAGE_SIZE)

/* HELPER FUNCTIONS AND DEFINITIONS */

// Whether to use custom exponentiation function or not.
#define CUSTOM_EXP

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
__hyperOp__ void SwiGLU(__Op32, __Op32, __Op32, __Op32, __Op32);

/* SMD DECLARATIONS */

__SMD__ smd_End = {.arity = 1, .ann = ANN_JOIN, .fptr = (__HyOpFunc)End};
__SMD__ smd_SwiGLU = {.arity = 5, .ann = ANN_NONE, .fptr = (__HyOpFunc)SwiGLU};

/* =============================================================================
 * STATIC BUFFER POINTERS
 * (Captured from host buffers at swiglu_start() entry. Flat-indexed since
 *  OpenCL kernel arguments arrive as raw mat_t* rather than fixed-size
 *  multi-dimensional arrays.)
 *
 * NOTE: SwiGLU has no intra-kernel scratch buffer — it's a single
 * elementwise stage (sigmoid(gate) * gate * up), so unlike rms.c/softmax.c
 * there is no temp[]/exp_vec[]-style device-internal global required.
 * =============================================================================*/
static mat_t *g_up;
static mat_t *g_gate;
static mat_t *g_output;

#define IDX_BSH(b,s,i)  (((b) * SEQ_LEN + (s)) * HIDDEN_SIZE + (i))

/* HYPEROP FUNCTION DEFINITIONS */

/// @brief Perform SwiGLU.
/// @param batch_id Batch index.
/// @param seq_id Sequence index.
/// @param base Base index.
/// @param bound Bound index.
/// @param endFr Context Memory Event Frame of End hyperOp.
__hyperOp__ void SwiGLU(__Op32 batch_id, __Op32 seq_id, __Op32 base,
                        __Op32 bound, __Op32 endFr) {
  // Indices.
  int b = (int)batch_id.i32;
  int s = (int)seq_id.i32;

  // Perform SwiGLU for the corresponding Up and Gate elements.
  for(int i = (int)base.i32; i < (int)bound.i32; i++) {
    float gate_val = g_gate[IDX_BSH(b,s,i)];
    float up_val = g_up[IDX_BSH(b,s,i)];
    float sigmoid = 1.0f / (1.0f + EXP(-gate_val));
    float silu = gate_val * sigmoid;
    g_output[IDX_BSH(b,s,i)] = silu * up_val;
  }

  // Signal current hyperOp completion to the End hyperOp.
  __sync(endFr.cmAddr, -1);

  // Delete current HyperOp.
  __fDelete(re_getSelfID());
}

/// End HyperOp.
__hyperOp__ void End() {
    re_println("End HyperOp — SwiGLU complete.\n");
    CE_FLUSH_ALL();
    LLC_FLUSH_ALL();
    __sync(_endKernelEventId, -1, memory_order_relaxed);
}

/* =============================================================================
 * swiglu_start - FPGA ENTRY POINT
 * =============================================================================*/
__kernel void swiglu_start(mat_t *up,
                           mat_t *gate,
                           mat_t *output)
{
    /* Capture host buffer pointers for use by asynchronous HyperOps. */
    g_up     = up;
    g_gate   = gate;
    g_output = output;

#ifdef CUSTOM_EXP
  re_println("Custom Function Used.");
#endif

    /* Create End HyperOp. */
    __CMAddr EndFr = __createInst(&smd_End);

    /* Set synchronisation wait count for End HyperOp Frame. */
    __sync(re_opAddr(EndFr, 15), BATCH_SIZE * SEQ_LEN * USAGE_SIZE);

    /* Initialise hyperOps. */
    for(int b = 0; b < BATCH_SIZE; b++) {
        for(int s = 0; s < SEQ_LEN; s++) {
            /* Create SwiGLU HyperOps. */
            for(int i = 0; i < USAGE_SIZE; i++) {
                /* Create SwiGLU HyperOp frame. */
                __CMAddr SwiGLUFr = __fAllocWithCe(1, i % CE_USE_NUM);
                __fBind(SwiGLUFr, &smd_SwiGLU);

                /* Pass operands to the SwiGLU HyperOp Frames. */
                __writeCM(re_opAddr(SwiGLUFr, 0), b);                     // batch_id
                __writeCM(re_opAddr(SwiGLUFr, 1), s);                     // seq_id
                __writeCM(re_opAddr(SwiGLUFr, 2), i * INC_SIZE);          // base
                __writeCM(re_opAddr(SwiGLUFr, 3), (i + 1) * INC_SIZE);    // bound
                __writeCM(re_opAddr(SwiGLUFr, 4), re_opAddr(EndFr, 15));  // endFr
            }
        }
    }
}