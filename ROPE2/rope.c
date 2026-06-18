#include <math.h>
#include "rope.h"
#include "redefine.h"

#define PASS 0
#define FAIL 1

#define CUSTOM_TRIG 1

#ifdef CUSTOM_TRIG
#define PI 3.14159265358979323846f
#define TWO_PI (2.0f * PI)

static float wrap_angle(float x) {
    while (x > PI)  x -= TWO_PI;
    while (x < -PI) x += TWO_PI;
    return x;
}

float my_sinf(float x) {
    x = wrap_angle(x);
    float x2 = x * x;
    return x * (1.0f - x2/6.0f + (x2*x2)/120.0f - (x2*x2*x2)/5040.0f);
}

float my_cosf(float x) {
    return my_sinf(x + PI/2.0f);
}
#endif

#if CUSTOM_TRIG
#define SIN(x) my_sinf(x)
#define COS(x) my_cosf(x)
#else
#define SIN(x) sinf(x)
#define COS(x) cosf(x)
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
 * SHARED BUFFERS
 * =============================================================================*/
__attribute__((aligned(64))) mat_t Q_in[N_Q_HEADS][HEAD_DIM];
__attribute__((aligned(64))) mat_t K_in[N_KV_HEADS][HEAD_DIM];
__attribute__((aligned(64))) mat_t Q_out[N_Q_HEADS][HEAD_DIM];
__attribute__((aligned(64))) mat_t K_out[N_KV_HEADS][HEAD_DIM];
__attribute__((aligned(64))) mat_t inv_freq[HALF];
__attribute__((aligned(64))) mat_t cos_cache[HALF];
__attribute__((aligned(64))) mat_t sin_cache[HALF];

int position; 

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
 * KERNEL MAIN --- START HYPEROP
 * =============================================================================*/
__kernel void __REDEFINE_main() {

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

/* =============================================================================
 * INVFREQ HYPEROP FUNCTION
 * =============================================================================*/
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

/* =============================================================================
 * COSCACHE HYPEROP FUNCTION
 * =============================================================================*/
__hyperOp__ void CosCacheOp(__Op32 ropeFr) {

    for (int i = 0; i < HALF; i++) {
        float angle = inv_freq[i] * (float)position;
        cos_cache[i] = COS(angle);
    }
    /* Signal RoPELaunch: cos_cache[] is ready. */
    __sync(ropeFr.cmAddr, -1);
}

/* =============================================================================
 * SINCACHE HYPEROP FUNCTION
 * =============================================================================*/
__hyperOp__ void SinCacheOp(__Op32 ropeFr) {

    for (int i = 0; i < HALF; i++) {
        float angle = inv_freq[i] * (float)position;
        sin_cache[i] = SIN(angle);
    }
    /* Signal RoPELaunch: sin_cache[] is ready. */
    __sync(ropeFr.cmAddr, -1);
}

/* =============================================================================
 * ROPELAUNCH HYPEROP FUNCTION
 * =============================================================================*/
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

/* =============================================================================
 * QROPE HYPEROP FUNCTION
 * =============================================================================*/
__hyperOp__ void QRopeOp(__Op32 head_id, __Op32 endFr) {
    int h = (int)head_id.i32;

    for (int i = 0; i < HALF; i++) {
        float c = cos_cache[i];
        float s = sin_cache[i];
        float a = Q_in[h][i];
        float b = Q_in[h][i + HALF];
        Q_out[h][i]        = a * c - b * s;
        Q_out[h][i + HALF] = b * c + a * s;
    }

    /* Pass-through: dimensions beyond ROTARY_DIM are not rotated. */
    for (int i = ROTARY_DIM; i < HEAD_DIM; i++) {
        Q_out[h][i] = Q_in[h][i];
    }

    /* Signal End: this Q head is done. */
    __sync(endFr.cmAddr, -1);
}

/* =============================================================================
 * KROPE HYPEROP FUNCTION
 * =============================================================================*/
__hyperOp__ void KRopeOp(__Op32 head_id, __Op32 endFr) {
    int h = (int)head_id.i32;

    for (int i = 0; i < HALF; i++) {
        float c = cos_cache[i];
        float s = sin_cache[i];
        float a = K_in[h][i];
        float b = K_in[h][i + HALF];
        K_out[h][i]        = a * c - b * s;
        K_out[h][i + HALF] = b * c + a * s;
    }

    /* Pass-through: dimensions beyond ROTARY_DIM are not rotated. */
    for (int i = ROTARY_DIM; i < HEAD_DIM; i++) {
        K_out[h][i] = K_in[h][i];
    }

    /* Signal End: this K head is done. */
    __sync(endFr.cmAddr, -1);
}

/* =============================================================================
 * END HYPEROP FUNCTION
 * =============================================================================*/
__hyperOp__ void End() {
    re_println("End HyperOp — RoPE complete.\n");
    re_sigEndOfKernel(PASS);
}
