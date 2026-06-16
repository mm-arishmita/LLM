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
 *  build time — same approach as wsum.c redefining HEAD_DIM/ATTN_HEADS/etc.
 *  Must match the values in common_defs.h used by the host.)
 * =============================================================================*/

/* Epsilon for avoiding zero RMS value. */
#define RMS_EPSILON 1e-5f

/* Hidden dimension size. */
#define HIDDEN_SIZE 3072

/* Batch size. */
#define BATCH_SIZE 1

/* Sequence length. */
#define SEQ_LEN 1

/* Number of Compute Elements to be used for processing. */
#define CE_USE_NUM 4

/* Amount of parallelisation. */
#define USAGE_SIZE 4

/* Number of hidden vector elements to be processed by 1 CE. */
#define INC_SIZE (HIDDEN_SIZE / USAGE_SIZE)

/* Floating point type used for all tensors. */
typedef float mat_t;

float custom_sqrtf(float val) {
    float result;

    __asm__ volatile (
        "fsqrt.s %0, %1"
        : "=f" (result)  // Output operand (%0)
        : "f" (val)   // Input operand (%1)
    );
    return result;
}

/* =============================================================================
 * HYPEROP FUNCTION DECLARATIONS
 * =============================================================================*/

__hyperOp__ void End();
__hyperOp__ void SkipAndSquare(__Op32, __Op32, __Op32, __Op32, __Op32);
__hyperOp__ void RMS(__Op32, __Op32, __Op32, __Op32);
__hyperOp__ void RMSNorm(__Op32, __Op32, __Op32, __Op32, __Op32, __Op32);

/* =============================================================================
 * SMD (Static Metadata) DECLARATIONS
 * =============================================================================*/
__SMD__ smd_End = {.arity = 1, .ann = ANN_JOIN, .fptr = (__HyOpFunc)End};
__SMD__ smd_SkipAndSquare = {
    .arity = 5, .ann = ANN_NONE, .fptr = (__HyOpFunc)SkipAndSquare};
__SMD__ smd_RMS = {.arity = 5, .ann = ANN_JOIN, .fptr = (__HyOpFunc)RMS};
__SMD__ smd_RMSNorm = {
    .arity = 6, .ann = ANN_NONE, .fptr = (__HyOpFunc)RMSNorm};

/* =============================================================================
 * STATIC BUFFER POINTERS
 * (Captured from host buffers at rms_start() entry. Flat-indexed since
 *  OpenCL kernel arguments arrive as raw mat_t* rather than fixed-size
 *  multi-dimensional arrays.)
 * =============================================================================*/
static mat_t *g_input;
static mat_t *g_hidden;
static mat_t *g_weights;
static mat_t *g_output;

#define IDX_BSH(b,s,i)  (((b) * SEQ_LEN + (s)) * HIDDEN_SIZE + (i))
#define IDX_H(i)        (i)

/* =============================================================================
 * SCRATCH BUFFERS (device-internal only — never written/read by host)
 * =============================================================================*/
__attribute__((aligned(64))) mat_t temp[BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE];
__attribute__((aligned(64))) mat_t residual[BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE];

/* =============================================================================
 * HYPEROP FUNCTION DEFINITIONS
 * =============================================================================*/

/// @brief Perform residual connection and calculate squared hidden vector.
/// @param batch_id Batch index.
/// @param seq_id Sequence index.
/// @param base Base index.
/// @param bound Bound index.
/// @param rmsFr Context Memory Event Frame of RMS hyperOp.
__hyperOp__ void SkipAndSquare(__Op32 batch_id, __Op32 seq_id, __Op32 base,
                               __Op32 bound, __Op32 rmsFr) {
  int batch = (int)batch_id.i32;
  int seq = (int)seq_id.i32;

  /* Perform Residual Connection (i.e. Skip), then Square each element. */
  for (int i = (int)base.i32; i < (int)bound.i32; i++) {
    residual[IDX_BSH(batch, seq, i)] =
        g_hidden[IDX_BSH(batch, seq, i)] + g_input[IDX_BSH(batch, seq, i)];
    temp[IDX_BSH(batch, seq, i)] =
        residual[IDX_BSH(batch, seq, i)] * residual[IDX_BSH(batch, seq, i)];
  }

  /* Signal current hyperOp completion to the RMS hyperOp. */
  __sync(rmsFr.cmAddr, -1);

  /* Delete current HyperOp. */
  __fDelete(re_getSelfID());
}

/// @brief Calculate RMS.
/// @param batch_id Batch index.
/// @param seq_id Sequence index.
/// @param epsilon Epsilon for avoiding zero RMS value.
/// @param endFr Context Memory Event Frame of End hyperOp.
__hyperOp__ void RMS(__Op32 batch_id, __Op32 seq_id, __Op32 epsilon,
                     __Op32 endFr) {
  int batch = (int)batch_id.i32;
  int seq = (int)seq_id.i32;

  /* Sum all elements. */
  float sq_sum = 0.0f;
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    sq_sum += temp[IDX_BSH(batch, seq, i)];
  }

  /* Compute RMS. */
  float inv_n = 1.0f / HIDDEN_SIZE;
  float ms = (sq_sum * inv_n) + epsilon.f32;
  float inv_rms = 1.0f / custom_sqrtf(ms);

  /* Create RMSNorm HyperOps. */
  for (int i = 0; i < USAGE_SIZE; i++) {
    __CMAddr RMSNormFr = __fAllocWithCe(1, (i / INC_SIZE) % CE_USE_NUM);
    __fBind(RMSNormFr, &smd_RMSNorm);

    __writeCM(re_opAddr(RMSNormFr, 0), batch);               /* batch_id */
    __writeCM(re_opAddr(RMSNormFr, 1), seq);                 /* seq_id */
    __writeCM(re_opAddr(RMSNormFr, 2), i * INC_SIZE);        /* base */
    __writeCM(re_opAddr(RMSNormFr, 3), (i + 1) * INC_SIZE);  /* bound */
    __writeCM(re_opAddr(RMSNormFr, 4), inv_rms);             /* rms_scale */
    __writeCM(re_opAddr(RMSNormFr, 5), endFr.cmAddr);        /* endFr */
  }
}

/// @brief Calculate RMSNorm.
/// @param batch_id Batch index.
/// @param seq_id Sequence index.
/// @param base Base index.
/// @param bound Bound index.
/// @param rms_scale Scale factor.
/// @param endFr Context Memory Event Frame of End hyperOp.
__hyperOp__ void RMSNorm(__Op32 batch_id, __Op32 seq_id, __Op32 base,
                         __Op32 bound, __Op32 rms_scale, __Op32 endFr) {
  int batch = (int)batch_id.i32;
  int seq = (int)seq_id.i32;

  /* Perform RMSNorm on each element. */
  for (int i = (int)base.i32; i < (int)bound.i32; i++) {
    g_output[IDX_BSH(batch, seq, i)] =
        g_weights[IDX_H(i)] * residual[IDX_BSH(batch, seq, i)] * rms_scale.f32;
  }

  /* Decrement synchronisation wait count for End HyperOp. */
  __sync(endFr.cmAddr, -1);

  /* Delete current HyperOp. */
  __fDelete(re_getSelfID());
}

/// End HyperOp.
__hyperOp__ void End() {
    re_println("End HyperOp — RMSNorm complete.\n");
    CE_FLUSH_ALL();
    LLC_FLUSH_ALL();
    __sync(_endKernelEventId, -1, memory_order_relaxed);
}

/* =============================================================================
 * rms_start - FPGA ENTRY POINT
 * =============================================================================*/
__kernel void rms_start(mat_t *input,
                        mat_t *hidden,
                        mat_t *weights,
                        mat_t *output)
{
    /* Capture host buffer pointers for use by asynchronous HyperOps. */
    g_input   = input;
    g_hidden  = hidden;
    g_weights = weights;
    g_output  = output;

    /* Create End HyperOp. */
    __CMAddr EndFr = __createInst(&smd_End);

    /* Set synchronisation wait count for End HyperOp Frame. */
    __sync(re_opAddr(EndFr, 15), BATCH_SIZE * SEQ_LEN * USAGE_SIZE);

    /* Initialise hyperOps. */
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            /* Create RMS HyperOp. */
            __CMAddr RMSFr = __createInst(&smd_RMS);

            /* Set synchronisation wait count for RMS HyperOp Frame. */
            __sync(re_opAddr(RMSFr, 15), USAGE_SIZE);

            /* Pass operands to the RMS HyperOp Frames. */
            __writeCM(re_opAddr(RMSFr, 0), b);                     /* batch_id */
            __writeCM(re_opAddr(RMSFr, 1), s);                     /* seq_id */
            __writeCM(re_opAddr(RMSFr, 2), RMS_EPSILON);           /* epsilon */
            __writeCM(re_opAddr(RMSFr, 3), re_opAddr(EndFr, 15));  /* endFr */

            /* Create SkipAndSquare HyperOps. */
            for (int i = 0; i < USAGE_SIZE; i++) {
                /* Create SkipAndSquare HyperOp frame. */
                __CMAddr SkipAndSquareFr = __fAllocWithCe(1, i % CE_USE_NUM);
                __fBind(SkipAndSquareFr, &smd_SkipAndSquare);

                /* Pass operands to the SkipAndSquare HyperOp Frames. */
                __writeCM(re_opAddr(SkipAndSquareFr, 0), b);             /* batch_id */
                __writeCM(re_opAddr(SkipAndSquareFr, 1), s);             /* seq_id */
                __writeCM(re_opAddr(SkipAndSquareFr, 2), i * INC_SIZE);  /* base */
                __writeCM(re_opAddr(SkipAndSquareFr, 3), (i + 1) * INC_SIZE);  /* bound */
                __writeCM(re_opAddr(SkipAndSquareFr, 4),
                          re_opAddr(RMSFr, 15));  /* rmsFr */
            }
        }
    }
}