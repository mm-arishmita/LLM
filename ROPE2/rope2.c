#include <math.h>
#include "rope.h"
#include "redefine.h"

#define PASS 0
#define FAIL 1

//#define CUSTOM_TRIG 1

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
typedef union {
    float f;
    unsigned int u;
} pow_float_bits;
static float frexp_custom(float x, int *exp_out) {
    pow_float_bits fb;
    fb.f = x;
    int raw_exp = (int)((fb.u >> 23) & 0xFF);
    int e = raw_exp - 126;                        /* gives m in [1.0, 2.0) */
    fb.u = (fb.u & 0x807FFFFFu) | (126u << 23);    /* force exponent field to represent [1,2) */
    *exp_out = e;
    return fb.f;
}
#define POW_LN2 0.6931471805599453f

static float ln_custom(float x) {
    if (x <= 0.0f) {
        /* Undefined for x <= 0; return a large negative sentinel
         * (shouldn't occur for RoPE's fixed positive base). */
        return -1e30f;
    }
 
    int e;
    float m = frexp_custom(x, &e);     /* x = m * 2^e, m in [1,2) */
 
    float s = (m - 1.0f) / (m + 1.0f); /* s in [0, 1/3) */
    float s2 = s * s;
    float series = 1.0f + s2 * (0.33333333f +
                    s2 * (0.2f +
                    s2 * (0.14285714f +
                    s2 * (0.11111111f +
                    s2 * (0.09090909f)))));
    float ln_m = 2.0f * s * series;
 
    return ln_m + (float)e * POW_LN2;
}
 
#define POW_LOG2E 1.4426950408889634f  /* 1 / ln(2) */

static float exp2_poly(float f) {
    return 1.0f + f * (0.69314718f +
           f * (0.24022651f +
           f * (0.05550411f +
           f * (0.00961813f +
           f * (0.00133336f +
           f * (0.00015401f +
           f * (0.00001525f)))))));
}
static float exp_custom(float x) {
    float y = x * POW_LOG2E;
    float y_round = (y >= 0.0f) ? (float)(long)(y + 0.5f) : (float)(long)(y - 0.5f);
    float f = y - y_round;             /* f in [-0.5, 0.5] */
 
    float p = exp2_poly(f);
    int n = (int)y_round;
    pow_float_bits fb;
    fb.u = (unsigned int)((n + 127) << 23);
 
    return p * fb.f;
}

float pow_lim(float base, float exponent) {
    if (base == 0.0f) {
        return (exponent == 0.0f) ? 1.0f : 0.0f;
    }
    return exp_custom(exponent * ln_custom(base));
}
#endif


#if CUSTOM_TRIG
#define SIN(x) sin_lim(x)
#define COS(x) cos_lim(x)
#define POW(a, b) pow_lim(a, b)
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
        inv_freq[i] = 1.0f / POW(ROPE_THETA, exponent);
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