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
 * (Redefined here rather than relying on rope.h resolving at OpenCL build
 *  time — same approach as wsum.c/rms.c/softmax.c/swiglu.c. Must match the
 *  values in rope.h used by the host.)
 * =============================================================================*/

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

/* =============================================================================
 * HYPEROP FUNCTION DECLARATIONS
 * =============================================================================*/
__hyperOp__ void End();
__hyperOp__ void InvFreqOp(__Op32, __Op32);
__hyperOp__ void CosCacheOp(__Op32);
__hyperOp__ void SinCacheOp(__Op32);
__hyperOp__ void RoPELaunch(__Op32);
__hyperOp__ void QRopeOp(__Op32, __Op32);
__hyperOp__ void KRopeOp(__Op32, __Op32);

/* =============================================================================
 * SMD (Static Metadata) DECLARATIONS
 * =============================================================================*/
__SMD__ smd_End = {.arity = 1, .ann = ANN_JOIN, .fptr  = (__HyOpFunc)End};

__SMD__ smd_InvFreqOp = {.arity = 2, .ann = ANN_NONE, .fptr  = (__HyOpFunc)InvFreqOp};

__SMD__ smd_CosCacheOp = {.arity = 2, .ann = ANN_JOIN, .fptr  = (__HyOpFunc)CosCacheOp};

__SMD__ smd_SinCacheOp = {.arity = 2, .ann = ANN_JOIN, .fptr  = (__HyOpFunc)SinCacheOp};

__SMD__ smd_RoPELaunch = {.arity = 2, .ann  = ANN_JOIN, .fptr  = (__HyOpFunc)RoPELaunch};

__SMD__ smd_QRopeOp = {.arity = 2, .ann  = ANN_NONE, .fptr  = (__HyOpFunc)QRopeOp};

__SMD__ smd_KRopeOp = {.arity = 2, .ann   = ANN_NONE, .fptr  = (__HyOpFunc)KRopeOp};

/* =============================================================================
 * STATIC BUFFER POINTERS
 * (Captured from host buffers at rope_start() entry. Flat-indexed since
 *  OpenCL kernel arguments arrive as raw mat_t* rather than fixed-size
 *  multi-dimensional arrays.)
 * =============================================================================*/
static mat_t *g_Q_in;
static mat_t *g_K_in;
static mat_t *g_Q_out;
static mat_t *g_K_out;

#define IDX_Q(h,d)   ((h) * HEAD_DIM + (d))
#define IDX_K(h,d)   ((h) * HEAD_DIM + (d))

/* =============================================================================
 * SCRATCH BUFFERS (device-internal only — never written/read by host)
 * inv_freq[], cos_cache[], sin_cache[] are computed entirely on-device from
 * `position`, never loaded from file, never read back by the host.
 * =============================================================================*/
__attribute__((aligned(64))) mat_t inv_freq[HALF];
__attribute__((aligned(64))) mat_t cos_cache[HALF];
__attribute__((aligned(64))) mat_t sin_cache[HALF];

/* =============================================================================
 * `position` — passed in as a plain scalar kernel argument (not a cl_mem
 * buffer) and captured into this kernel-local global at rope_start() entry,
 * so CosCacheOp/SinCacheOp can keep reading it exactly as before.
 * =============================================================================*/
static int position;

/* =============================================================================
 * HYPEROP FRAMES
 * =============================================================================*/
__CMAddr InvFreqFr;
__CMAddr CosCacheFr;
__CMAddr SinCacheFr;
__CMAddr RoPELaunchFr;
__CMAddr QRopeFr[N_Q_HEADS];
__CMAddr KRopeFr[N_KV_HEADS];

/* =============================================================================
 * HYPEROP FUNCTION DEFINITIONS
 * =============================================================================*/

/// @brief Compute inverse frequency table.
/// @param cosFr Context Memory Event Frame of CosCache hyperOp.
/// @param sinFr Context Memory Event Frame of SinCache hyperOp.
__hyperOp__ void InvFreqOp(__Op32 cosFr, __Op32 sinFr) {

    for (int i = 0; i < HALF; i++) {
        float exponent = (2.0f * (float)i) / (float)ROTARY_DIM;
        inv_freq[i] = 1.0f / powf(ROPE_THETA, exponent);
    }

    /* Signal CosCache: inv_freq[] is ready. */
    __sync(cosFr.cmAddr, -1);

    /* Signal SinCache: same inv_freq[] is ready.               */
    __sync(sinFr.cmAddr, -1);
}

/// @brief Compute cosine cache.
/// @param ropeFr Context Memory Event Frame of RoPELaunch hyperOp.
__hyperOp__ void CosCacheOp(__Op32 ropeFr) {

    for (int i = 0; i < HALF; i++) {
        float angle = inv_freq[i] * (float)position;
        cos_cache[i] = cosf(angle);
    }
    /* Signal RoPELaunch: cos_cache[] is ready. */
    __sync(ropeFr.cmAddr, -1);
}

/// @brief Compute sine cache.
/// @param ropeFr Context Memory Event Frame of RoPELaunch hyperOp.
__hyperOp__ void SinCacheOp(__Op32 ropeFr) {

    for (int i = 0; i < HALF; i++) {
        float angle = inv_freq[i] * (float)position;
        sin_cache[i] = sinf(angle);
    }
    /* Signal RoPELaunch: sin_cache[] is ready. */
    __sync(ropeFr.cmAddr, -1);
}

/// @brief Launch QRope and KRope hyperOps for all heads.
/// @param endFr Context Memory Event Frame of End hyperOp.
__hyperOp__ void RoPELaunch(__Op32 endFr) {

  /* ── Create and fire QRope frames — one per Q head ───────────────────── */
    for (int h = 0; h < N_Q_HEADS; h++) {
        QRopeFr[h] = __createInst(&smd_QRopeOp);

        __writeCM(re_opAddr(QRopeFr[h], 0), h);
        __writeCM(re_opAddr(QRopeFr[h], 1), endFr);
    }

    /* ── Create and fire KRope frames — one per KV head ───────────────────── */
    for (int h = 0; h < N_KV_HEADS; h++) {
        KRopeFr[h] = __createInst(&smd_KRopeOp);

        __writeCM(re_opAddr(KRopeFr[h], 0), h);
        __writeCM(re_opAddr(KRopeFr[h], 1), endFr);
    }
}

/// @brief Rotate one Q head.
/// @param head_id Head index.
/// @param endFr Context Memory Event Frame of End hyperOp.
__hyperOp__ void QRopeOp(__Op32 head_id, __Op32 endFr) {
    int h = (int)head_id.i32;

    for (int i = 0; i < HALF; i++) {
        float c = cos_cache[i];
        float s = sin_cache[i];
        float a = g_Q_in[IDX_Q(h,i)];
        float b = g_Q_in[IDX_Q(h,i + HALF)];
        g_Q_out[IDX_Q(h,i)]        = a * c - b * s;
        g_Q_out[IDX_Q(h,i + HALF)] = b * c + a * s;
    }

    /* Pass-through: dimensions beyond ROTARY_DIM are not rotated. */
    for (int i = ROTARY_DIM; i < HEAD_DIM; i++) {
        g_Q_out[IDX_Q(h,i)] = g_Q_in[IDX_Q(h,i)];
    }

    /* Signal End: this Q head is done. */
    __sync(endFr.cmAddr, -1);
}

/// @brief Rotate one K head.
/// @param head_id Head index.
/// @param endFr Context Memory Event Frame of End hyperOp.
__hyperOp__ void KRopeOp(__Op32 head_id, __Op32 endFr) {
    int h = (int)head_id.i32;

    for (int i = 0; i < HALF; i++) {
        float c = cos_cache[i];
        float s = sin_cache[i];
        float a = g_K_in[IDX_K(h,i)];
        float b = g_K_in[IDX_K(h,i + HALF)];
        g_K_out[IDX_K(h,i)]        = a * c - b * s;
        g_K_out[IDX_K(h,i + HALF)] = b * c + a * s;
    }

    /* Pass-through: dimensions beyond ROTARY_DIM are not rotated. */
    for (int i = ROTARY_DIM; i < HEAD_DIM; i++) {
        g_K_out[IDX_K(h,i)] = g_K_in[IDX_K(h,i)];
    }

    /* Signal End: this K head is done. */
    __sync(endFr.cmAddr, -1);
}

/// End HyperOp.
__hyperOp__ void End() {
    re_println("End HyperOp — RoPE complete.\n");
    CE_FLUSH_ALL();
    LLC_FLUSH_ALL();
    __sync(_endKernelEventId, -1, memory_order_relaxed);
}

/* =============================================================================
 * rope_start - FPGA ENTRY POINT
 * =============================================================================*/
__kernel void rope_start(mat_t *Q_in,
                         mat_t *K_in,
                         mat_t *Q_out,
                         mat_t *K_out,
                         int   pos)
{
    /* Capture host buffer pointers for use by asynchronous HyperOps. */
    g_Q_in   = Q_in;
    g_K_in   = K_in;
    g_Q_out  = Q_out;
    g_K_out  = K_out;

    /* Capture the scalar position argument into the kernel-local global,
     * so CosCacheOp/SinCacheOp can keep reading `position` unchanged. */
    position = pos;

    __CMAddr EndFr = __createInst(&smd_End);
    __sync(re_opAddr(EndFr, 15), N_Q_HEADS + N_KV_HEADS);

    RoPELaunchFr = __createInst(&smd_RoPELaunch);
    __sync(re_opAddr(RoPELaunchFr, 15), 2);   /* 2 = CosCache + SinCache      */

    CosCacheFr = __createInst(&smd_CosCacheOp);
    __sync(re_opAddr(CosCacheFr, 15), 1);

    SinCacheFr = __createInst(&smd_SinCacheOp);
    __sync(re_opAddr(SinCacheFr, 15), 1);

    InvFreqFr = __createInst(&smd_InvFreqOp);

    //OPERANDS FOR Invfreqop//
    __writeCM(re_opAddr(InvFreqFr, 0), re_opAddr(CosCacheFr, 15));
    __writeCM(re_opAddr(InvFreqFr, 1), re_opAddr(SinCacheFr, 15));

    //OPERANDS FOR Coscacheop//
    __writeCM(re_opAddr(CosCacheFr, 0), re_opAddr(RoPELaunchFr, 15));

    //OPERANDS FOR Sincacheop//
    __writeCM(re_opAddr(SinCacheFr, 0), re_opAddr(RoPELaunchFr, 15));

     //OPERANDS FOR Ropelaunchop//
    __writeCM(re_opAddr(RoPELaunchFr, 0), re_opAddr(EndFr, 15));
}