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

#define CUSTOM_TRIG 1

///CUSTOM SINF COSF//
#ifdef CUSTOM_TRIG
#define TWO_OVER_PI 0.6366197723675814f
#define PI_2_A 1.5703125f
#define PI_2_B 0.0004838267923332751f
#define PI_2_C 2.5632829192545614e-12f

static float round_nearest(float x) {
    if (x >= 0.0f) {
        return (float)(long)(x + 0.5f);
    } else {
        return (float)(long)(x - 0.5f);
    }
}
static void range_reduce_quad(float x, float *r_out, int *quadrant) {
    float k = round_nearest(x * TWO_OVER_PI);
 
    float r = x - k * PI_2_A;
    r = r - k * PI_2_B;
    r = r - k * PI_2_C;
 
    *r_out = r;
 
    long ki = (long)k;
    int q = (int)(ki % 4);
    if (q < 0) q += 4;
    *quadrant = q;
}
static float sin_small(float r) {
    float r2 = r * r;
    return r * (0.9999966f +
           r2 * (-0.16664824f +
           r2 * (0.00830629f +
           r2 * (-0.00018363f))));
}
static float cos_small(float r) {
    float r2 = r * r;
    return 0.99999934f +
           r2 * (-0.49999125f +
           r2 * (0.04166204f +
           r2 * (-0.00133527f +
           r2 * (0.00002314f))));
}

float sin_lim(float x) {
    float r;
    int q;
    range_reduce_quad(x, &r, &q);
 
    switch (q) {
        case 0: return  sin_small(r);
        case 1: return  cos_small(r);
        case 2: return -sin_small(r);
        default: return -cos_small(r);  /* q == 3 */
    }
}
 
float cos_lim(float x) {
    float r;
    int q;
    range_reduce_quad(x, &r, &q);
 
    switch (q) {
        case 0: return  cos_small(r);
        case 1: return -sin_small(r);
        case 2: return -cos_small(r);
        default: return  sin_small(r);  /* q == 3 */
    }
}
#endif

#ifdef CUSTOM_TRIG
// Approximate log and exp with higher accuracy
static float my_logf(float x) {
    // Use a polynomial approximation around 1
    float y = (x - 1.0f) / (x + 1.0f);
    float y2 = y * y;
    // Padé-style expansion for better accuracy
    return 2.0f * (y + (y2*y)/3.0f + (y2*y2*y)/5.0f + (y2*y2*y2*y)/7.0f);
}

static float my_expf(float x) {
    // 7-term Taylor expansion
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x3 * x;
    float x5 = x4 * x;
    float x6 = x5 * x;
    return 1.0f + x + x2/2.0f + x3/6.0f + x4/24.0f + x5/120.0f + x6/720.0f;
}

// Custom powf: a^b
float my_powf(float a, float b) {
    return my_expf(b * my_logf(a));
}
#endif


#if CUSTOM_TRIG
#define SIN(x) sin_lim(x)
#define COS(x) cos_lim(x)
#define POW(a, b) my_powf(a, b)
#else
#define SIN(x) sinf(x)
#define COS(x) cosf(x)
#define POW(a, b) powf(a, b)
#endif
 

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
        inv_freq[i] = 1.0f / POW(ROPE_THETA, exponent);
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
        cos_cache[i] = COS(angle);
    }
    /* Signal RoPELaunch: cos_cache[] is ready. */
    __sync(ropeFr.cmAddr, -1);
}

/// @brief Compute sine cache.
/// @param ropeFr Context Memory Event Frame of RoPELaunch hyperOp.
__hyperOp__ void SinCacheOp(__Op32 ropeFr) {

    for (int i = 0; i < HALF; i++) {
        float angle = inv_freq[i] * (float)position;
        sin_cache[i] = SIN(angle);
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