/* =============================================================================
 * Host Code for RoPE Kernel
 * =============================================================================*/
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 220
#include "CL/opencl.h"

#include "rope.h"
#include "common.h"

#define PROGRAM_FILE "rope.c"

/* =============================================================================
 * TARGET DEVICE
 * =============================================================================*/
#if defined(__FUNCTSIM__)
DeviceName targetDevice = rfsim_hyperop;   /* functional simulator */
#else
DeviceName targetDevice = rsim_hyperop;    /* RTL simulator / FPGA */
#endif

/* =============================================================================
 * HOST BUFFER DECLARATIONS
 *
 * NOTE: inv_freq[], cos_cache[], sin_cache[] from the kernel source are NOT
 * declared here. They are intra-kernel scratch buffers — computed entirely
 * on-device from `position`, never loaded from file and never read back by
 * the host. Therefore they require no corresponding cl_mem buffer or
 * clSetKernelArg entry.
 *
 * `position` is a scalar (not a buffer) and is passed via clSetKernelArg
 * directly as an int, not as a cl_mem.
 * =============================================================================*/

static mat_t Q_in  [N_Q_HEADS ][HEAD_DIM];
static mat_t K_in  [N_KV_HEADS][HEAD_DIM];
static mat_t Q_out [N_Q_HEADS ][HEAD_DIM];
static mat_t K_out [N_KV_HEADS][HEAD_DIM];

/* =============================================================================
 * HOST-ONLY BUFFERS (reference data, unchanged from FSim host code)
 * =============================================================================*/
static mat_t ref_Q_out[N_Q_HEADS][HEAD_DIM];    /* Golden Q_out from extractor   */
static mat_t ref_K_out[N_KV_HEADS][HEAD_DIM];   /* Golden K_out from extractor   */
static mat_t manual_Q_out[N_Q_HEADS][HEAD_DIM]; /* Re-computed reference on host */
static mat_t manual_K_out[N_KV_HEADS][HEAD_DIM];

static int position;

#ifndef DEBUG_CODE
#define DEBUG_CODE 0
#endif

/* =============================================================================
 * TIMING / PROFILING HELPERS
 * =============================================================================*/
static double host_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((double)ts.tv_sec * 1000.0) + ((double)ts.tv_nsec / 1.0e6);
}

/* Returns elapsed device time for an event in milliseconds (0.0 on failure) */
static double profile_event_elapsed_ms(cl_event event)
{
    cl_ulong start = 0;
    cl_ulong end   = 0;

    if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(start), &start, NULL) != CL_SUCCESS ||
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(end), &end, NULL) != CL_SUCCESS) {
        return 0.0;
    }
    return (double)(end - start) / 1.0e6;
}

static void write_timing_profile(const char *label,
                                  double write_ms,
                                  double kernel_ms,
                                  double read_ms,
                                  double total_ms)
{
    FILE *timing_log = fopen("timing_profile.txt", "w");
    if (timing_log == NULL) {
        printf("Unable to write timing_profile.txt\n");
        return;
    }
    fprintf(timing_log, "===== TIMING LOGS: %s =====\n", label);
    fprintf(timing_log, "Write time (Q_in,K_in upload) : %.6f ms\n", write_ms);
    fprintf(timing_log, "Kernel execution time         : %.6f ms\n", kernel_ms);
    fprintf(timing_log, "Read time (Q_out,K_out)        : %.6f ms\n", read_ms);
    fprintf(timing_log, "Profiled OpenCL time          : %.6f ms\n",
            write_ms + kernel_ms + read_ms);
    fprintf(timing_log, "Total host wall time          : %.6f ms\n", total_ms);
    fprintf(timing_log, "===============================\n");
    fclose(timing_log);
    printf("Timing profile written to timing_profile.txt\n");
}

/* =============================================================================
 * BINARY LOADER
 * Reads exactly `count` float32 values from `path` into `buf`.
 * Returns 0 on success, -1 on failure.
 * =============================================================================*/
static int load_f32_bin(const char *path, float *buf, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open '%s'\n", path);
        return -1;
    }
    size_t got = fread(buf, sizeof(float), count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "[ERROR] '%s': expected %zu floats, got %zu\n",
                path, count, got);
        return -1;
    }
    return 0;
}

/* =============================================================================
 * MANUAL REFERENCE IMPLEMENTATION
 * Mirrors apply_rope_single() in model.c and manual_rope() in the original
 * rope test exactly. Unchanged from the FSim host code.
 * =============================================================================*/
static void manual_rope(void) {
    float ref_inv_freq[HALF];
    float ref_cos[HALF];
    float ref_sin[HALF];

    /* Step 1: build inv_freq, cos, sin */
    for (int i = 0; i < HALF; i++) {
        float exponent   = (2.0f * (float)i) / (float)ROTARY_DIM;
        ref_inv_freq[i]  = 1.0f / powf(ROPE_THETA, exponent);
        float angle      = ref_inv_freq[i] * (float)position;
        ref_cos[i]       = cosf(angle);
        ref_sin[i]       = sinf(angle);
    }

    /* Step 2: rotate Q heads */
    for (int h = 0; h < N_Q_HEADS; h++) {
        for (int i = 0; i < HALF; i++) {
            float c = ref_cos[i], s = ref_sin[i];
            float a = Q_in[h][i], b = Q_in[h][i + HALF];
            manual_Q_out[h][i]        = a * c - b * s;
            manual_Q_out[h][i + HALF] = b * c + a * s;
        }
        /* Pass-through beyond ROTARY_DIM */
        for (int i = ROTARY_DIM; i < HEAD_DIM; i++)
            manual_Q_out[h][i] = Q_in[h][i];

#if DEBUG_CODE
        printf("manual_Q_out[%d][0] = %f\n", h, manual_Q_out[h][0]);
#endif
    }

    /* Step 3: rotate K heads */
    for (int h = 0; h < N_KV_HEADS; h++) {
        for (int i = 0; i < HALF; i++) {
            float c = ref_cos[i], s = ref_sin[i];
            float a = K_in[h][i], b = K_in[h][i + HALF];
            manual_K_out[h][i]        = a * c - b * s;
            manual_K_out[h][i + HALF] = b * c + a * s;
        }
        /* Pass-through beyond ROTARY_DIM */
        for (int i = ROTARY_DIM; i < HEAD_DIM; i++)
            manual_K_out[h][i] = K_in[h][i];

#if DEBUG_CODE
        printf("manual_K_out[%d][0] = %f\n", h, manual_K_out[h][0]);
#endif
    }
}

/* =============================================================================
 * COMPARE HELPERS (unchanged from FSim host code)
 * =============================================================================*/
static int compare_Q(const mat_t a[N_Q_HEADS][HEAD_DIM],
                     const mat_t b[N_Q_HEADS][HEAD_DIM],
                     float tol,
                     const char *label_a,
                     const char *label_b) {
    int ok = 1;
    for (int h = 0; h < N_Q_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float diff = fabsf(a[h][d] - b[h][d]);
            if (diff > tol) {
                ok = 0;
                //printf("[MISMATCH] Q h=%d d=%d : %s=%.8f  %s=%.8f  diff=%.2e\n",
                       //h, d, label_a, a[h][d], label_b, b[h][d], diff);
            }
        }
    }
    return ok;
}

static int compare_K(const mat_t a[N_KV_HEADS][HEAD_DIM],
                     const mat_t b[N_KV_HEADS][HEAD_DIM],
                     float tol,
                     const char *label_a,
                     const char *label_b) {
    int ok = 1;
    for (int h = 0; h < N_KV_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float diff = fabsf(a[h][d] - b[h][d]);
            if (diff > tol) {
                ok = 0;
                //printf("[MISMATCH] K h=%d d=%d : %s=%.8f  %s=%.8f  diff=%.2e\n",
                       //h, d, label_a, a[h][d], label_b, b[h][d], diff);
            }
        }
    }
    return ok;
}

/* =============================================================================
 * run_rope()
 * =============================================================================*/
static int run_rope(void)
{
    cl_int error = 0;
    double total_start_ms = host_time_ms();

    /* ------------------------------------------------------------------
     * OpenCL init
     * ------------------------------------------------------------------ */
    cl_device_id     deviceId  = create_device(targetDevice);

    cl_context       contextId = clCreateContext(NULL, 1, &deviceId,
                                                 NULL, NULL, &error);
    checkErr(error, "clCreateContext");

    /* Enable profiling so we can measure kernel / read / write timings */
    const cl_queue_properties queue_props[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,
        0};

    cl_command_queue cmdqId    = clCreateCommandQueueWithProperties(
                                     contextId, deviceId, queue_props, &error);
    checkErr(error, "clCreateCommandQueueWithProperties");

    /* ------------------------------------------------------------------
     * Load and build kernel program
     * ------------------------------------------------------------------ */
    FILE *fh = fopen(PROGRAM_FILE, "r");
    if (!fh) { perror("Couldn't find " PROGRAM_FILE); exit(EXIT_FAILURE); }

    fseek(fh, 0, SEEK_END);
    size_t prog_size = (size_t)ftell(fh);
    rewind(fh);

    char *prog_buf = (char *)malloc(prog_size + 1);
    prog_buf[prog_size] = '\0';
    fread(prog_buf, sizeof(char), prog_size, fh);
    fclose(fh);

    cl_program programId = clCreateProgramWithSource(
                               contextId, 1,
                               (const char **)&prog_buf, &prog_size,
                               &error);
    if (error < 0) { perror("clCreateProgramWithSource"); exit(EXIT_FAILURE); }
    free(prog_buf);

    const char *compile_options = {""};
    error = clBuildProgram(programId, 1, &deviceId, compile_options, NULL, NULL);
    checkErr(error, "clBuildProgram");

    /* "rope_start"  ↔  __kernel void rope_start(...) in rope.c */
    cl_kernel kernelId = clCreateKernel(programId, "rope_start", &error);
    checkErr(error, "clCreateKernel rope_start");

    /* ------------------------------------------------------------------
     * Allocate device buffers
     *
     *  dev_Q_in   READ_ONLY   → uploaded via write event
     *  dev_K_in   READ_ONLY   → uploaded via write event
     *  dev_Q_out  WRITE_ONLY  → read back after kernel
     *  dev_K_out  WRITE_ONLY  → read back after kernel
     *
     * `position` is passed as a plain scalar kernel argument (not a
     * cl_mem buffer) via clSetKernelArg.
     * ------------------------------------------------------------------ */
    size_t sz_Q_in  = sizeof(mat_t) * N_Q_HEADS  * HEAD_DIM;
    size_t sz_K_in  = sizeof(mat_t) * N_KV_HEADS * HEAD_DIM;
    size_t sz_Q_out = sizeof(mat_t) * N_Q_HEADS  * HEAD_DIM;
    size_t sz_K_out = sizeof(mat_t) * N_KV_HEADS * HEAD_DIM;

    cl_mem dev_Q_in = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY,
                                  sz_Q_in, NULL, &error);
    checkErr(error, "clCreateBuffer dev_Q_in");

    cl_mem dev_K_in = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY,
                                  sz_K_in, NULL, &error);
    checkErr(error, "clCreateBuffer dev_K_in");

    cl_mem dev_Q_out = clCreateBuffer(contextId,
                                  CL_MEM_WRITE_ONLY,
                                  sz_Q_out, NULL, &error);
    checkErr(error, "clCreateBuffer dev_Q_out");

    cl_mem dev_K_out = clCreateBuffer(contextId,
                                  CL_MEM_WRITE_ONLY,
                                  sz_K_out, NULL, &error);
    checkErr(error, "clCreateBuffer dev_K_out");

    /* ------------------------------------------------------------------
     * Upload Q_in and K_in (timed via profiling events)
     * ------------------------------------------------------------------ */
    cl_event write_Q_in_event = NULL;
    cl_event write_K_in_event = NULL;

    error = clEnqueueWriteBuffer(cmdqId, dev_Q_in, CL_FALSE, 0,
                                 sz_Q_in, Q_in, 0, NULL, &write_Q_in_event);
    checkErr(error, "clEnqueueWriteBuffer dev_Q_in");

    error = clEnqueueWriteBuffer(cmdqId, dev_K_in, CL_FALSE, 0,
                                 sz_K_in, K_in, 0, NULL, &write_K_in_event);
    checkErr(error, "clEnqueueWriteBuffer dev_K_in");

    error = clFinish(cmdqId);
    checkErr(error, "clFinish (after writes)");

    double write_ms = profile_event_elapsed_ms(write_Q_in_event)
                     + profile_event_elapsed_ms(write_K_in_event);

    clReleaseEvent(write_Q_in_event);
    clReleaseEvent(write_K_in_event);

    /* ------------------------------------------------------------------
     * Set kernel arguments  →  rope_start(Q_in, K_in, Q_out, K_out, position)
     * ------------------------------------------------------------------ */
    error = clSetKernelArg(kernelId, 0, sizeof(cl_mem), &dev_Q_in);
    checkErr(error, "clSetKernelArg dev_Q_in");

    error = clSetKernelArg(kernelId, 1, sizeof(cl_mem), &dev_K_in);
    checkErr(error, "clSetKernelArg dev_K_in");

    error = clSetKernelArg(kernelId, 2, sizeof(cl_mem), &dev_Q_out);
    checkErr(error, "clSetKernelArg dev_Q_out");

    error = clSetKernelArg(kernelId, 3, sizeof(cl_mem), &dev_K_out);
    checkErr(error, "clSetKernelArg dev_K_out");

    error = clSetKernelArg(kernelId, 4, sizeof(int), &position);
    checkErr(error, "clSetKernelArg position");

    /* ------------------------------------------------------------------
     * Launch kernel (timed via profiling event)
     * ------------------------------------------------------------------ */
    size_t global[1] = {4};
    size_t local[1]  = {1};

    cl_event kernel_event = NULL;

    error = clEnqueueNDRangeKernel(cmdqId, kernelId, 1,
                                   NULL, global, local,
                                   0, NULL, &kernel_event);
    checkErr(error, "clEnqueueNDRangeKernel");

    error = clFinish(cmdqId);
    checkErr(error, "clFinish");

    double kernel_ms = profile_event_elapsed_ms(kernel_event);
    clReleaseEvent(kernel_event);

    /* ------------------------------------------------------------------
     * Read Q_out and K_out back to host (timed via profiling events)
     * ------------------------------------------------------------------ */
    cl_event read_Q_out_event = NULL;
    cl_event read_K_out_event = NULL;

    error = clEnqueueReadBuffer(cmdqId, dev_Q_out,
                                CL_FALSE, 0, sz_Q_out,
                                Q_out,
                                0, NULL, &read_Q_out_event);
    checkErr(error, "clEnqueueReadBuffer dev_Q_out");

    error = clEnqueueReadBuffer(cmdqId, dev_K_out,
                                CL_TRUE, 0, sz_K_out,
                                K_out,
                                0, NULL, &read_K_out_event);
    checkErr(error, "clEnqueueReadBuffer dev_K_out");

    double read_ms = profile_event_elapsed_ms(read_Q_out_event)
                    + profile_event_elapsed_ms(read_K_out_event);

    clReleaseEvent(read_Q_out_event);
    clReleaseEvent(read_K_out_event);

    double total_ms = host_time_ms() - total_start_ms;

    /* ------------------------------------------------------------------
     * Print and persist timing profile
     * ------------------------------------------------------------------ */
    printf("\n===== TIMING LOGS: RoPE =====\n");
    printf("Write time (Q_in,K_in upload) : %.6f ms\n", write_ms);
    printf("Kernel execution time         : %.6f ms\n", kernel_ms);
    printf("Read time (Q_out,K_out)        : %.6f ms\n", read_ms);
    printf("Profiled OpenCL time          : %.6f ms\n",
           write_ms + kernel_ms + read_ms);
    printf("Total host wall time          : %.6f ms\n", total_ms);
    printf("===============================\n");

    write_timing_profile("RoPE", write_ms, kernel_ms, read_ms, total_ms);

    /* ------------------------------------------------------------------
     * Release all OpenCL objects
     * ------------------------------------------------------------------ */
    clReleaseMemObject(dev_Q_in);
    clReleaseMemObject(dev_K_in);
    clReleaseMemObject(dev_Q_out);
    clReleaseMemObject(dev_K_out);

    clReleaseKernel(kernelId);
    clReleaseProgram(programId);
    clReleaseCommandQueue(cmdqId);
    clReleaseContext(contextId);

    cl_platform_id platformId;
    error = clGetDeviceInfo(deviceId, CL_DEVICE_PLATFORM,
                            sizeof(platformId), &platformId, NULL);
    checkErr(error, "clGetDeviceInfo");

    if (error == CL_SUCCESS) {
        error = clUnloadPlatformCompiler(platformId);
        checkErr(error, "clUnloadPlatformCompiler");
    }

    return CL_SUCCESS;
}

/* =============================================================================
 * MAIN
 * =============================================================================*/
int main(int argc, char *argv[]) {

    /* -------------------------------------------------------------------------
     * 0. Parse optional dump-file paths from command line.
     *    Defaults match the extractor's naming convention for layer 1, pos 0.
     * -------------------------------------------------------------------------*/
    const char *q_in_path  = (argc > 1) ? argv[1] : "dump_L1_pos0_Q_in.bin";
    const char *k_in_path  = (argc > 2) ? argv[2] : "dump_L1_pos0_K_in.bin";
    const char *q_out_path = (argc > 3) ? argv[3] : "dump_L1_pos0_Q_out.bin";
    const char *k_out_path = (argc > 4) ? argv[4] : "dump_L1_pos0_K_out.bin";

    /* position: prefer command-line arg, else fall back to compile-time default */
#ifdef POSITION
    position = POSITION;
#else
    position = 0;
#endif
    if (argc > 5) position = atoi(argv[5]);

    printf("=== Phi-4-mini Layer-%d RoPE Kernel Test (FPGA / OpenCL) ===\n", 1);
    printf("N_Q_HEADS=%d  N_KV_HEADS=%d  HEAD_DIM=%d\n",
           N_Q_HEADS, N_KV_HEADS, HEAD_DIM);
    printf("ROTARY_DIM=%d  HALF=%d  ROPE_THETA=%.0f  position=%d\n\n",
           ROTARY_DIM, HALF, (float)ROPE_THETA, position);

    /* -------------------------------------------------------------------------
     * 1. Load Q_in  [N_Q_HEADS][HEAD_DIM]
     * -------------------------------------------------------------------------*/
    printf("Loading Q_in  from '%s'...\n", q_in_path);
    if (load_f32_bin(q_in_path, (float *)Q_in,
                     (size_t)N_Q_HEADS * HEAD_DIM) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 2. Load K_in  [N_KV_HEADS][HEAD_DIM]
     * -------------------------------------------------------------------------*/
    printf("Loading K_in  from '%s'...\n", k_in_path);
    if (load_f32_bin(k_in_path, (float *)K_in,
                     (size_t)N_KV_HEADS * HEAD_DIM) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 3. Load golden Q_out  [N_Q_HEADS][HEAD_DIM]
     * -------------------------------------------------------------------------*/
    printf("Loading ref Q_out from '%s'...\n", q_out_path);
    if (load_f32_bin(q_out_path, (float *)ref_Q_out,
                     (size_t)N_Q_HEADS * HEAD_DIM) != 0) return 1;

    /* -------------------------------------------------------------------------
     * 4. Load golden K_out  [N_KV_HEADS][HEAD_DIM]
     * -------------------------------------------------------------------------*/
    printf("Loading ref K_out from '%s'...\n", k_out_path);
    if (load_f32_bin(k_out_path, (float *)ref_K_out,
                     (size_t)N_KV_HEADS * HEAD_DIM) != 0) return 1;

    printf("All inputs loaded successfully.\n\n");

    /* -------------------------------------------------------------------------
     * 5. Re-compute RoPE manually on host (second sanity check).
     * -------------------------------------------------------------------------*/
    manual_rope();

    /* -------------------------------------------------------------------------
     * 6. Check 1: manual host recompute vs extractor reference.
     * -------------------------------------------------------------------------*/
    printf("--- Check 1: manual_out vs extractor ref_out (tol=1e-4) ---\n");
    int check1_Q = compare_Q(manual_Q_out, ref_Q_out, 1e-4f, "manual_Q", "ref_Q");
    int check1_K = compare_K(manual_K_out, ref_K_out, 1e-4f, "manual_K", "ref_K");
    int check1   = check1_Q && check1_K;
    if (check1)
        printf("  PASS: manual recompute matches extractor reference.\n\n");
    else
        printf("  FAIL: layout or formula mismatch — fix before running chip.\n\n");

#if DEBUG_CODE
    printf("--- Q_in (head 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  Q_in[0][%d] = %f\n", d, Q_in[0][d]);
    printf("--- K_in (head 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  K_in[0][%d] = %f\n", d, K_in[0][d]);
    printf("--- ref_Q_out (head 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  ref_Q_out[0][%d] = %f\n", d, ref_Q_out[0][d]);
    printf("--- ref_K_out (head 0) ---\n");
    for (int d = 0; d < HEAD_DIM; d++)
        printf("  ref_K_out[0][%d] = %f\n", d, ref_K_out[0][d]);
#endif

    /* -------------------------------------------------------------------------
     * 7. Launch the HyperOps kernel on the FPGA via OpenCL.
     * -------------------------------------------------------------------------*/
    printf("--- Launching HyperOps kernel (OpenCL / FPGA) ---\n");
    int rc = run_rope();
    if (rc != CL_SUCCESS) {
        printf("Kernel execution failed.\n");
        return 1;
    }
    printf("Kernel execution complete.\n\n");

    /* -------------------------------------------------------------------------
     * 8. Check 2: chip Q_out / K_out vs extractor reference.
     * -------------------------------------------------------------------------*/
    printf("--- Check 2: chip Q_out/K_out vs extractor ref (tol=1e-4) ---\n");
    int check2_Q = compare_Q(Q_out, ref_Q_out, 1e-6f, "chip_Q", "ref_Q");
    int check2_K = compare_K(K_out, ref_K_out, 1e-6f, "chip_K", "ref_K");
    int check2   = check2_Q && check2_K;

    /* -------------------------------------------------------------------------
     * 9. Check 3: chip Q_out / K_out vs manual host recompute.
     * -------------------------------------------------------------------------*/
    printf("--- Check 3: chip Q_out/K_out vs manual recompute (tol=1e-4) ---\n");
    int check3_Q = compare_Q(Q_out, manual_Q_out, 1e-6f, "chip_Q", "manual_Q");
    int check3_K = compare_K(K_out, manual_K_out, 1e-6f, "chip_K", "manual_K");
    int check3   = check3_Q && check3_K;

    /* -------------------------------------------------------------------------
     * 10. Final verdict.
     * -------------------------------------------------------------------------*/
    printf("\n=== Result ===\n");
    if (check1 && check2 && check3) {
        printf("RoPE computation SUCCESSFUL.\n");
    } else {
        if (!check1) printf("  [FAIL] Check 1: manual vs ref    — data load/layout issue.\n");
        if (!check2) printf("  [FAIL] Check 2: chip vs ref      — HyperOps computation error.\n");
        if (!check3) printf("  [FAIL] Check 3: chip vs manual   — HyperOps computation error.\n");
        printf("RoPE computation UNSUCCESSFUL.\n");
    }

    return (check1 && check2 && check3) ? 0 : 1;
}