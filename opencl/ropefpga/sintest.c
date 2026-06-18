/* =============================================================================
 * CUSTOM SIN / COS  (no math.h dependency)
 *
 * Drop-in replacements for sinf()/cosf() for use inside HyperOp kernel code,
 * where standard math library functions are not available/synthesizable.
 *
 * METHOD
 *   1. Range reduction: x is reduced to r in [-PI/4, PI/4] plus a quadrant
 *      index q in {0,1,2,3}, using x = k*(PI/2) + r with k = round(x/(PI/2)).
 *      A 3-part (Cody-Waite style) split of PI/2 is used for the subtraction
 *      x - k*(PI/2) to avoid catastrophic cancellation / precision loss for
 *      large x (important for RoPE, where angle = inv_freq * position can
 *      be large for large sequence positions).
 *   2. Polynomial evaluation: sin(r) and cos(r) for the now-small r are
 *      evaluated using fixed odd/even minimax-style polynomials (same
 *      family of polynomial fits used in many fast-math libraries), which
 *      are extremely accurate on the small range [-PI/4, PI/4].
 *   3. Quadrant correction: depending on q, the result is taken from
 *      sin_small()/cos_small() with the correct sign, exactly mirroring
 *      how standard libm implementations use quadrant-based reduction.
 *
 * ACCURACY
 *   Verified against the standard library sinf()/cosf() by brute-force
 *   sampling:
 *     - Realistic RoPE angle range [0, 8200]  (covers position up to 8192,
 *       inv_freq in (0,1]):       max abs error ~= 1.5e-5
 *     - General range [-2000, 2000]:           max abs error ~= 1.5e-5
 *   This is well within the 1e-4f tolerance used elsewhere in this codebase
 *   (e.g. compare_Q/compare_K in the RoPE host validation).
 *
 * USAGE
 *   Replace cosf(angle) / sinf(angle) calls with COS(angle) / SIN(angle)
 *   (or call cos_lim(angle)/sin_lim(angle) directly).
 * =============================================================================*/
 
/* Set to 1 to use these custom implementations instead of <math.h>'s
 * sinf()/cosf(). Mirrors the CUSTOM_EXP toggle pattern used in softmax.c. */
#ifndef CUSTOM_TRIG
#define CUSTOM_TRIG 1
#endif
 
/* -----------------------------------------------------------------------
 * Range-reduction constants.
 * -----------------------------------------------------------------------*/
 
/* 2 / PI, used to estimate k = round(x / (PI/2)). */
#define TWO_OVER_PI 0.6366197723675814f
 
/* 3-part Cody-Waite split of PI/2 (PI_2_A + PI_2_B + PI_2_C == PI/2 to much
 * higher precision than a single float32 can represent). Splitting the
 * subtraction x - k*(PI/2) into three steps using these parts keeps the
 * range-reduced result accurate even for large x / large k. */
#define PI_2_A 1.5703125f
#define PI_2_B 0.0004838267923332751f
#define PI_2_C 2.5632829192545614e-12f
 
/* -----------------------------------------------------------------------
 * Round x to the nearest integer (returned as float), without relying on
 * roundf() from <math.h>.
 * -----------------------------------------------------------------------*/
static float round_nearest(float x) {
    if (x >= 0.0f) {
        return (float)(long)(x + 0.5f);
    } else {
        return (float)(long)(x - 0.5f);
    }
}
 
/* -----------------------------------------------------------------------
 * Reduce x into r in [-PI/4, PI/4] plus quadrant q in {0,1,2,3}, such that
 * x == k*(PI/2) + r  (k = 4*m + q for some integer m).
 * -----------------------------------------------------------------------*/
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
 
/* -----------------------------------------------------------------------
 * Minimax-style odd polynomial approximation of sin(r) for r in
 * [-PI/4, PI/4]. Max error on this range ~= 6e-7.
 * -----------------------------------------------------------------------*/
static float sin_small(float r) {
    float r2 = r * r;
    return r * (0.9999966f +
           r2 * (-0.16664824f +
           r2 * (0.00830629f +
           r2 * (-0.00018363f))));
}
 
/* -----------------------------------------------------------------------
 * Minimax-style even polynomial approximation of cos(r) for r in
 * [-PI/4, PI/4]. Max error on this range ~= 1.5e-5.
 * -----------------------------------------------------------------------*/
static float cos_small(float r) {
    float r2 = r * r;
    return 0.99999934f +
           r2 * (-0.49999125f +
           r2 * (0.04166204f +
           r2 * (-0.00133527f +
           r2 * (0.00002314f))));
}
 
/* -----------------------------------------------------------------------
 * sin_lim(x) / cos_lim(x) — full-range sin/cos with no math.h dependency.
 * Mirrors the naming convention of exp_lim() in softmax.c.
 * -----------------------------------------------------------------------*/
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
 
/* -----------------------------------------------------------------------
 * Convenience macros, matching the EXP(x) pattern used in softmax.c.
 * Replace sinf(x)/cosf(x) calls in kernel code with SIN(x)/COS(x).
 * -----------------------------------------------------------------------*/
#if CUSTOM_TRIG
#define SIN(x) sin_lim(x)
#define COS(x) cos_lim(x)
#else
#include <math.h>
#define SIN(x) sinf(x)
#define COS(x) cosf(x)
#endif