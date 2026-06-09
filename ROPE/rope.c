#include <math.h>
#include "rope.h"
#include "redefine.h"

#define PASS 0
#define FAIL 1

/* =============================================================================
 * COMPUTATION OVERVIEW
 * =============================================================================
 *
 *  Step 1 — InvFreq  (1 hyperop, fires immediately):
 *      inv_freq[i] = 1.0 / pow(ROPE_THETA, (2*i) / ROTARY_DIM)
 *      for i = 0 .. HALF-1   where HALF = ROTARY_DIM / 2
 *
 *  Step 2 — CosCache, SinCache  (1 hyperop each, wait for InvFreq):
 *      cos_cache[i] = cos(inv_freq[i] * position)
 *      sin_cache[i] = sin(inv_freq[i] * position)
 *
 *  Step 3 — RoPE launcher  (1 hyperop, waits for Cos + Sin):
 *      Fires QRope[h] for h = 0..N_Q_HEADS-1
 *      Fires KRope[h] for h = 0..N_KV_HEADS-1
 *
 *  Step 4 — QRope[h], KRope[h]  (N_Q_HEADS + N_KV_HEADS hyperops):
 *      For each dimension pair i in [0, HALF):
 *          a = head[i],   b = head[i + HALF]
 *          head[i]        = a * cos_cache[i] - b * sin_cache[i]
 *          head[i + HALF] = b * cos_cache[i] + a * sin_cache[i]
 *
 *  Step 5 — End  (ANN_JOIN, waits for all QRope + KRope):
 *      Q_out[h][d] and K_out[h][d] are valid for all h, d.
 *
 * =============================================================================
 * DEPENDENCY GRAPH
 *
 *           InvFreq
 *           /     \
 *      CosCache  SinCache
 *           \     /
 *            RoPE               ← launcher hyperop
 *          /  ...  \
 *    QRope[0]    KRope[0]
 *    QRope[1]    KRope[1]
 *       ...         ...
 *    QRope[Hq-1] KRope[Hk-1]
 *          \     /
 *            End
 *
 * =============================================================================
 * SYNC COUNTS
 *
 *   InvFreq  : 0  (ANN_NONE — fires when slots written)
 *   CosCache : 1  (waits for InvFreq)
 *   SinCache : 1  (waits for InvFreq)
 *   RoPE     : 2  (waits for CosCache AND SinCache)
 *   QRope[h] : 0  (ANN_NONE — RoPE launcher fires them)
 *   KRope[h] : 0  (ANN_NONE — RoPE launcher fires them)
 *   End      : N_Q_HEADS + N_KV_HEADS
 * =============================================================================*/

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
 *
 *  INPUTS  (filled by caller before kernel fires):
 *      Q_in[h][d]   — query vectors,  shape [N_Q_HEADS][HEAD_DIM]
 *      K_in[h][d]   — key vectors,    shape [N_KV_HEADS][HEAD_DIM]
 *      position     — current token position (scalar)
 *
 *  SCRATCH (written by InvFreq/Cos/Sin, read by QRope/KRope):
 *      inv_freq[i]  — inverse frequencies,  shape [HALF]
 *      cos_cache[i] — cosine values,        shape [HALF]
 *      sin_cache[i] — sine values,          shape [HALF]
 *
 *  OUTPUTS (written by QRope/KRope):
 *      Q_out[h][d]  — rotated queries, shape [N_Q_HEADS][HEAD_DIM]
 *      K_out[h][d]  — rotated keys,    shape [N_KV_HEADS][HEAD_DIM]
 * =============================================================================*/
__attribute__((aligned(64))) mat_t Q_in[N_Q_HEADS][HEAD_DIM];
__attribute__((aligned(64))) mat_t K_in[N_KV_HEADS][HEAD_DIM];
__attribute__((aligned(64))) mat_t Q_out[N_Q_HEADS][HEAD_DIM];
__attribute__((aligned(64))) mat_t K_out[N_KV_HEADS][HEAD_DIM];
__attribute__((aligned(64))) mat_t inv_freq[HALF];
__attribute__((aligned(64))) mat_t cos_cache[HALF];
__attribute__((aligned(64))) mat_t sin_cache[HALF];

int position;   /* token position — written by host before kernel launch */

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
 * KERNEL MAIN
 *
 * Launch sequence:
 *   1. Create End frame;          sync count = N_Q_HEADS + N_KV_HEADS.
 *   2. Create RoPELaunch frame;   sync count = 2  (Cos + Sin).
 *   3. Create CosCache frame;     sync count = 1  (InvFreq).
 *   4. Create SinCache frame;     sync count = 1  (InvFreq).
 *   5. Create InvFreq frame.
 *   6. Write InvFreq operand → fires immediately (ANN_NONE).
 *      InvFreq will signal CosCache and SinCache when done.
 *   7. Write CosCache + SinCache operands (blocked by sync count = 1).
 *   8. Write RoPELaunch operand   (blocked by sync count = 2).
 *
 *   QRope/KRope frames are created inside RoPELaunch hyperop,
 *   because we need End's CMAddr to pass to them — and End is created here.
 * =============================================================================*/
__kernel void __REDEFINE_main() {

    /* ── End frame ─────────────────────────────────────────────────────────── */
    __CMAddr EndFr = __createInst(&smd_End);
    __sync(re_opAddr(EndFr, 15), 4);

    /* ── RoPELaunch frame — waits for CosCache + SinCache ──────────────────── */
    RoPELaunchFr = __createInst(&smd_RoPELaunch);
    __sync(re_opAddr(RoPELaunchFr, 15), 2);   /* 2 = CosCache + SinCache      */

    /* ── CosCache frame — waits for InvFreq ────────────────────────────────── */
    CosCacheFr = __createInst(&smd_CosCacheOp);
    __sync(re_opAddr(CosCacheFr, 15), 1);

    /* ── SinCache frame — waits for InvFreq ────────────────────────────────── */
    SinCacheFr = __createInst(&smd_SinCacheOp);
    __sync(re_opAddr(SinCacheFr, 15), 1);

    /* ── InvFreq frame — fires immediately (ANN_NONE) ───────────────────────── */
    InvFreqFr = __createInst(&smd_InvFreqOp);
    /*   Slot 0 and 1: packed CMAddr of both CosCache and SinCache sync slots.
     *   InvFreq will signal both after computing inv_freq[].
     *   We pass CosCacheFr sync slot; InvFreq also signals SinCacheFr directly. */
    __writeCM(re_opAddr(InvFreqFr, 0), re_opAddr(CosCacheFr, 15));
    __writeCM(re_opAddr(InvFreqFr, 0), re_opAddr(SinCacheFr, 15));

    /* ── Write CosCache operand (blocked until InvFreq signals) ─────────────── */
    /*   Slot 0: sync addr of RoPELaunch — CosCache signals it when done.       */
    __writeCM(re_opAddr(CosCacheFr, 0), re_opAddr(RoPELaunchFr, 15));

    /* ── Write SinCache operand (blocked until InvFreq signals) ─────────────── */
    /*   Slot 0: sync addr of RoPELaunch — SinCache signals it when done.       */
    __writeCM(re_opAddr(SinCacheFr, 0), re_opAddr(RoPELaunchFr, 15));

    /* ── Write RoPELaunch operand (blocked until Cos + Sin signal) ──────────── */
    /*   Slot 0: CMAddr of End sync slot — passed through to QRope/KRope.       */
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
    __sync(re_opAddr(sinFr.cmAddr, 15), -1);
}

/* =============================================================================
 * COSCACHE HYPEROP  (ANN_JOIN — fires after InvFreq signals)
 *
 * Computes cosine cache for current token position.
 *   cos_cache[i] = cos(inv_freq[i] * position)
 *   for i = 0 .. HALF-1
 *
 * @param ropeFr  CMAddr of RoPELaunch's sync slot (slot 15).
 * =============================================================================*/
__hyperOp__ void CosCacheOp(__Op32 ropeFr) {

    for (int i = 0; i < HALF; i++) {
        float angle = inv_freq[i] * (float)position;
        cos_cache[i] = cosf(angle);
    }
    /* Signal RoPELaunch: cos_cache[] is ready. */
    __sync(ropeFr.cmAddr, -1);
}

/* =============================================================================
 * SINCACHE HYPEROP  (ANN_JOIN — fires after InvFreq signals)
 *
 * Computes sine cache for current token position.
 *   sin_cache[i] = sin(inv_freq[i] * position)
 *   for i = 0 .. HALF-1
 *
 * @param ropeFr  CMAddr of RoPELaunch's sync slot (slot 15).
 * =============================================================================*/
__hyperOp__ void SinCacheOp(__Op32 ropeFr) {

    for (int i = 0; i < HALF; i++) {
        float angle = inv_freq[i] * (float)position;
        sin_cache[i] = sinf(angle);
    }
    /* Signal RoPELaunch: sin_cache[] is ready. */
    __sync(ropeFr.cmAddr, -1);
}

/* =============================================================================
 * ROPELAUNCH HYPEROP  (ANN_JOIN — fires after CosCache + SinCache signal)
 *
 * Launcher: creates and fires all QRope and KRope hyperop frames.
 * Each head runs independently in parallel.
 *
 * @param endFr  CMAddr of End HyperOp's sync slot — passed to all Q/K heads.
 * =============================================================================*/
__hyperOp__ void RoPELaunch(__Op32 endFr) {

    /* ── Create and fire QRope frames — one per Q head ─────────────────────── */
    for (int h = 0; h < N_Q_HEADS; h++) {
        QRopeFr[h] = __createInst(&smd_QRopeOp);
        /* Slot layout:
         *   0 — head_id : Q head index h
         *   1 — endFr   : CMAddr of End sync slot                           */
        __writeCM(re_opAddr(QRopeFr[h], 0), h);
        __writeCM(re_opAddr(QRopeFr[h], 1), endFr);
    }

    /* ── Create and fire KRope frames — one per KV head ───────────────────── */
    for (int h = 0; h < N_KV_HEADS; h++) {
        KRopeFr[h] = __createInst(&smd_KRopeOp);
        /* Slot layout:
         *   0 — head_id : KV head index h
         *   1 — endFr   : CMAddr of End sync slot                           */
        __writeCM(re_opAddr(KRopeFr[h], 0), h);
        __writeCM(re_opAddr(KRopeFr[h], 1), endFr);
    }
}

/* =============================================================================
 * QROPE HYPEROP  (ANN_NONE — fired by RoPELaunch)
 *
 * Applies RoPE rotation to one Q head.
 * For each dimension pair i in [0, HALF):
 *   a = Q_in[h][i],        b = Q_in[h][i + HALF]
 *   Q_out[h][i]        = a * cos_cache[i] - b * sin_cache[i]
 *   Q_out[h][i + HALF] = b * cos_cache[i] + a * sin_cache[i]
 *
 * Dimensions beyond ROTARY_DIM are copied unchanged (pass-through).
 *
 * @param head_id  Q head index h.
 * @param endFr    CMAddr of End HyperOp's sync slot.
 * =============================================================================*/
__hyperOp__ void QRopeOp(__Op32 head_id, __Op32 endFr) {
    int h = (int)head_id.i32;

    /* Apply rotation to the rotary portion [0, ROTARY_DIM). */
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
 * KROPE HYPEROP  (ANN_NONE — fired by RoPELaunch)
 *
 * Applies RoPE rotation to one K head.
 * Same rotation formula as QRope — applied to K_in/K_out.
 *
 * @param head_id  KV head index h.
 * @param endFr    CMAddr of End HyperOp's sync slot.
 * =============================================================================*/
__hyperOp__ void KRopeOp(__Op32 head_id, __Op32 endFr) {
    int h = (int)head_id.i32;

    /* Apply rotation to the rotary portion [0, ROTARY_DIM). */
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
 * END HYPEROP  (ANN_JOIN)
 *
 * Fires after all N_Q_HEADS QRope + N_KV_HEADS KRope hyperops have signalled.
 * Q_out[h][d] and K_out[h][d] are valid for all h, d.
 * =============================================================================*/
__hyperOp__ void End() {
    re_println("End HyperOp — RoPE complete.\n");
    re_sigEndOfKernel(PASS);
}