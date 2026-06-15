/* =============================================================================
 * Host Code for Weighted Sum Kernel
 * =============================================================================*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 220
#include "CL/opencl.h"

#include "wsum.h"
#include "common.h"

#define PROGRAM_FILE "wsum.c"

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
 * =============================================================================*/

static mat_t Scores[ATTN_HEADS][SEQ_LEN];                  /* Softmax weights (input)    */
static mat_t V[SEQ_LEN][HEAD_DIM];             /* Value cache    (input)     */
//static mat_t partials_v[ATTN_HEADS][CE_USE_NUM][HEAD_DIM]; /* CE scratch                 */
static mat_t Ctx[ATTN_HEADS][HEAD_DIM];                    /* Output: context vectors    */
static mat_t manual_ctx[ATTN_HEADS][HEAD_DIM];

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
    fprintf(timing_log, "Write time (Scores,V upload) : %.6f ms\n", write_ms);
    fprintf(timing_log, "Kernel execution time   : %.6f ms\n", kernel_ms);
    fprintf(timing_log, "Read time (ctx)      : %.6f ms\n", read_ms);
    fprintf(timing_log, "Profiled OpenCL time    : %.6f ms\n",
            write_ms + kernel_ms + read_ms);
    fprintf(timing_log, "Total host wall time    : %.6f ms\n", total_ms);
    fprintf(timing_log, "===============================\n");
    fclose(timing_log);
    printf("Timing profile written to timing_profile.txt\n");
}


//#ifndef DEBUG_CODE
//#define DEBUG_CODE 0
//#endif


/* =============================================================================
 * MANUAL IMPLEMENTATION
 *
 * Reference weighted sum on the host CPU.
 *   manual_ctx[h][d] = Σ_{tok=0}^{SEQ_LEN-1}  Scores[h][tok] * V[h][tok][d]
 * =============================================================================*/
static void manual_weighted_sum() {
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0;
            for (int tok = 0; tok < SEQ_LEN; tok++) {
                acc += (float)Scores[h][tok] * (float)V[h][tok][d];
            }
            manual_ctx[h][d] = (float)acc;

//#if DEBUG_CODE
            //printf("manual_ctx[%d][%d] = %f\n", h, d, manual_ctx[h][d]);
//#endif
        }
    }
}

/* =============================================================================
 * run_wsum()
 * =============================================================================*/
static int run_wsum(void)
{
    cl_int error = 0;
    double total_start_ms = host_time_ms();

    /* ------------------------------------------------------------------
     * OpenCL init
     * FSim:  redefine_initialize(1, 1)
     * FPGA:  create_device → clCreateContext → clCreateCommandQueueWithProperties
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
     * FSim:  (none — kernel compiled separately, linked as __REDEFINE_main)
     * FPGA:  fopen → clCreateProgramWithSource → clBuildProgram → clCreateKernel
     *        (identical to cltest.c program loading block)
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

    /* "wsum_start"  ↔  __kernel void wsum_start(...) in wsum.c */
    cl_kernel kernelId = clCreateKernel(programId, "wsum_start", &error);
    checkErr(error, "clCreateKernel wsum_start");

    /* ------------------------------------------------------------------
     * Allocate device buffers
     *
     * NOTE: dev_scores / dev_V no longer use CL_MEM_COPY_HOST_PTR so that the
     * upload can be issued explicitly via clEnqueueWriteBuffer and timed
     * with a profiling event.
     *
     *  dev_Scores        READ_ONLY                  → uploaded via write event
     *  dev_V        READ_ONLY                  → uploaded via write event
     *  dev_partials READ_WRITE                  → device-internal scratch
     *  dev_ctx   WRITE_ONLY                  → read back after kernel
     * ------------------------------------------------------------------ */
    size_t sz_Scores   = sizeof(mat_t) * ATTN_HEADS * SEQ_LEN;
    size_t sz_V        = sizeof(mat_t) * SEQ_LEN    * HEAD_DIM;
    size_t sz_partials = sizeof(mat_t) * ATTN_HEADS * CE_USE_NUM * HEAD_DIM;
    size_t sz_ctx      = sizeof(mat_t) * ATTN_HEADS * HEAD_DIM;

    cl_mem dev_Q = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY,
                                  sz_Scores, NULL, &error);
    checkErr(error, "clCreateBuffer dev_Scores");

    cl_mem dev_K = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY,
                                  sz_V, NULL, &error);
    checkErr(error, "clCreateBuffer dev_V");

    cl_mem dev_partials = clCreateBuffer(contextId,
                                          CL_MEM_READ_WRITE,
                                          sz_partials, NULL, &error);
    checkErr(error, "clCreateBuffer dev_partials");

    cl_mem dev_Scores   = clCreateBuffer(contextId,
                                          CL_MEM_WRITE_ONLY,
                                          sz_ctx, NULL, &error);
    checkErr(error, "clCreateBuffer dev_ctx");

    /* ------------------------------------------------------------------
     * Upload Scores and V (timed via profiling events)
     * ------------------------------------------------------------------ */
    cl_event write_Scores_event = NULL;
    cl_event write_V_event = NULL;

    error = clEnqueueWriteBuffer(cmdqId, dev_Scores, CL_FALSE, 0,
                                 sz_Scores, Scores, 0, NULL, &write_Scores_event);
    checkErr(error, "clEnqueueWriteBuffer dev_Scores");

    error = clEnqueueWriteBuffer(cmdqId, dev_V, CL_FALSE, 0,
                                 sz_V, V, 0, NULL, &write_V_event);
    checkErr(error, "clEnqueueWriteBuffer dev_V");

    error = clFinish(cmdqId);
    checkErr(error, "clFinish (after writes)");

    double write_ms = profile_event_elapsed_ms(write_Scores_event)
                     + profile_event_elapsed_ms(write_V_event);

    clReleaseEvent(write_Scores_event);
    clReleaseEvent(write_V_event);

    /* ------------------------------------------------------------------
     * Set kernel arguments  →  att_start(Q, K, partials,scores)
     * ------------------------------------------------------------------ */
    error = clSetKernelArg(kernelId, 0, sizeof(cl_mem), &dev_Scores);
    checkErr(error, "clSetKernelArg dev_Scores");

    error = clSetKernelArg(kernelId, 1, sizeof(cl_mem), &dev_V);
    checkErr(error, "clSetKernelArg dev_V");

    error = clSetKernelArg(kernelId, 2, sizeof(cl_mem), &dev_partials);
    checkErr(error, "clSetKernelArg dev_partials");

    error = clSetKernelArg(kernelId, 3, sizeof(cl_mem), &dev_ctx);
    checkErr(error, "clSetKernelArg dev_ctx");

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
     * Read Scores back to host (timed via profiling event)
     * ------------------------------------------------------------------ */
    cl_event read_event = NULL;

    error = clEnqueueReadBuffer(cmdqId, dev_ctx,
                                CL_TRUE, 0, sz_ctx,
                                Ctx,
                                0, NULL, &read_event);
    checkErr(error, "clEnqueueReadBuffer dev_Scores");

    double read_ms = profile_event_elapsed_ms(read_event);
    clReleaseEvent(read_event);

    double total_ms = host_time_ms() - total_start_ms;

    /* ------------------------------------------------------------------
     * Validate device output against reference
     * ------------------------------------------------------------------ */
    int valid = 1;
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int d = 0; t < HEAD_DIM; d++) {
            if (Ctx[h][d] != manual_ctx[h][d]) {
                valid = 0;
                printf("[MISMATCH] Ctx[%d][%d]: chip=%f  ref=%f\n",
                       h, d, Ctx[h][d], manual_ctx[h][d]);
            }
        }
    }

    if (valid)
        printf("Weighted Sum computation successful.\n");
    else
        printf("Weighted Sum computation unsuccessful!\n");

    /* ------------------------------------------------------------------
     * Print and persist timing profile
     * ------------------------------------------------------------------ */
    printf("\n===== TIMING LOGS: AttentionScore =====\n");
    printf("Write time (Scores,V upload) : %.6f ms\n", write_ms);
    printf("Kernel execution time   : %.6f ms\n", kernel_ms);
    printf("Read time (Scores)      : %.6f ms\n", read_ms);
    printf("Profiled OpenCL time    : %.6f ms\n",
           write_ms + kernel_ms + read_ms);
    printf("Total host wall time    : %.6f ms\n", total_ms);
    printf("===============================\n");

    write_timing_profile("Weighted_Sum", write_ms, kernel_ms, read_ms, total_ms);

    /* ------------------------------------------------------------------
     * Release all OpenCL objects
     * ------------------------------------------------------------------ */
    clReleaseMemObject(dev_Scores);
    clReleaseMemObject(dev_V);
    clReleaseMemObject(dev_partials);
    clReleaseMemObject(dev_ctx);

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

    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(void) {

    printf("#####################################################\n");
    printf("######  TESTING WEIGHTED SUM (FPGA / OpenCL) ####\n");
    printf("#####################################################\n");

    /* -------------------------------------------------------------------------
     * 1. Initialise Scores (softmax weights).
     *    Each head gets a distinct pattern; rows are manually normalised
     *    so that Σ_tok Scores[h][tok] == 1  (mirrors softmax output).
     *
     *    Pattern:  raw[h][tok] = (h + 1) * (tok + 1)
     *    Normalise: Scores[h][tok] = raw[h][tok] / Σ_tok raw[h][tok]
     *    (different scale per head → distinct weighted sums per head)
     * -------------------------------------------------------------------------*/
    for (int h = 0; h < ATTN_HEADS; h++) {
        /* Compute raw unnormalised weights. */
        float row_sum = 0.0;
        for (int tok = 0; tok < SEQ_LEN; tok++) {
            Scores[h][tok] = (float)((h + 1) * (tok + 1));
            row_sum += Scores[h][tok];
        }
        /* Normalise so the row sums to 1. */
        for (int tok = 0; tok < SEQ_LEN; tok++) {
            Scores[h][tok] = (float)((float)Scores[h][tok] / row_sum);
        }
    }

    /* -------------------------------------------------------------------------
     * 2. Initialise V (value) buffers.
     *    Each (token, dim) gets a distinct pattern so heads produce
     *    different context vectors.
     *
     *    V[h][tok][d] = (tok % 2 == 0) ? (d % 2 == 0 ? 1.0f : 0.0f)   (alternating)
     *                                   : (float)(d + 1) / HEAD_DIM      (ramp)
     *    (mirrors K initialisation pattern from att_main.c)
     * -------------------------------------------------------------------------*/
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int tok = 0; tok < SEQ_LEN; tok++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                if (tok % 2 == 0) {
                    /* Even tokens: alternating 1/0 pattern. */
                    V[h][tok][d] = (d % 2 == 0) ? 1.0f : 0.0f;
                } else {
                    /* Odd tokens: linear ramp. */
                    V[h][tok][d] = (float)(d + 1) / (float)HEAD_DIM;
                }
            }
        }
    }

    /* -------------------------------------------------------------------------
     * 3. Manual weighted sum computation.
     * -------------------------------------------------------------------------*/
    manual_weighted_sum();

    /* -------------------------------------------------------------------------
     * 4.Run on FPGA via OpenCL.
     * -------------------------------------------------------------------------*/
     return run_wsum();
  
    
}
