/* =============================================================================
 * Host Code for Softmax Kernel
 * =============================================================================*/
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 220
#include "CL/opencl.h"

#include "common_defs.h"
#include "common.h"

#define PROGRAM_FILE "softmax.c"

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
 * NOTE: exp_vec[] from the kernel source is NOT declared here. It is an
 * intra-kernel scratch buffer (statically allocated as a global inside
 * softmax.c) used between the Exponent and Sum/Softmax hyperOp stages.
 * The host never writes to or reads from it, so it requires no
 * corresponding cl_mem buffer or clSetKernelArg entry.
 * =============================================================================*/

static mat_t score [BATCH_SIZE][HEAD_NUM][Q_LEN][K_LEN];
static mat_t output[BATCH_SIZE][HEAD_NUM][Q_LEN][K_LEN];

//#ifndef DEBUG_CODE
//define DEBUG_CODE 0
//#endif

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
    fprintf(timing_log, "Write time (score upload) : %.6f ms\n", write_ms);
    fprintf(timing_log, "Kernel execution time     : %.6f ms\n", kernel_ms);
    fprintf(timing_log, "Read time (output)        : %.6f ms\n", read_ms);
    fprintf(timing_log, "Profiled OpenCL time      : %.6f ms\n",
            write_ms + kernel_ms + read_ms);
    fprintf(timing_log, "Total host wall time      : %.6f ms\n", total_ms);
    fprintf(timing_log, "===============================\n");
    fclose(timing_log);
    printf("Timing profile written to timing_profile.txt\n");
}

/* =============================================================================
 * DEBUG-DUMP LOADING 
 * =============================================================================*/

float* load_vector_f32(const char* path, int expected_size) {
  FILE* f = fopen(path, "rb");
  if(!f) {
    perror("fopen failed");
    return NULL;
  }

  float* data = (float*)malloc(sizeof(float) * expected_size);
  if(!data) {
    fclose(f);
    return NULL;
  }

  size_t read = fread(data, sizeof(float), expected_size, f);
  fclose(f);

  if(read != (size_t)expected_size) {
    fprintf(stderr, "Size mismatch: expected %d, got %zu\n", expected_size,
            read);
    free(data);
    return NULL;
  }

  return data;
}

/* =============================================================================
 * run_softmax()
 * =============================================================================*/
static int run_softmax(float *dbg_output)
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

    /* "softmax_start"  ↔  __kernel void softmax_start(...) in softmax.c */
    cl_kernel kernelId = clCreateKernel(programId, "softmax_start", &error);
    checkErr(error, "clCreateKernel softmax_start");

    /* ------------------------------------------------------------------
     * Allocate device buffers
     *  dev_score    READ_ONLY   → uploaded via write event
     *  dev_output   WRITE_ONLY  → read back after kernel
     * ------------------------------------------------------------------ */
    size_t sz_score  = sizeof(mat_t) * BATCH_SIZE * HEAD_NUM * Q_LEN * K_LEN;
    size_t sz_output = sizeof(mat_t) * BATCH_SIZE * HEAD_NUM * Q_LEN * K_LEN;

    cl_mem dev_score = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY,
                                  sz_score, NULL, &error);
    checkErr(error, "clCreateBuffer dev_score");

    cl_mem dev_output = clCreateBuffer(contextId,
                                  CL_MEM_WRITE_ONLY,
                                  sz_output, NULL, &error);
    checkErr(error, "clCreateBuffer dev_output");

    /* ------------------------------------------------------------------
     * Upload score (timed via profiling event)
     * ------------------------------------------------------------------ */
    cl_event write_score_event = NULL;

    error = clEnqueueWriteBuffer(cmdqId, dev_score, CL_FALSE, 0,
                                 sz_score, score, 0, NULL, &write_score_event);
    checkErr(error, "clEnqueueWriteBuffer dev_score");

    error = clFinish(cmdqId);
    checkErr(error, "clFinish (after write)");

    double write_ms = profile_event_elapsed_ms(write_score_event);
    clReleaseEvent(write_score_event);

    /* ------------------------------------------------------------------
     * Set kernel arguments  →  softmax_start(score, output)
     * ------------------------------------------------------------------ */
    error = clSetKernelArg(kernelId, 0, sizeof(cl_mem), &dev_score);
    checkErr(error, "clSetKernelArg dev_score");

    error = clSetKernelArg(kernelId, 1, sizeof(cl_mem), &dev_output);
    checkErr(error, "clSetKernelArg dev_output");

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
     * Read output back to host (timed via profiling event)
     * ------------------------------------------------------------------ */
    cl_event read_event = NULL;

    error = clEnqueueReadBuffer(cmdqId, dev_output,
                                CL_TRUE, 0, sz_output,
                                output,
                                0, NULL, &read_event);
    checkErr(error, "clEnqueueReadBuffer dev_output");

    double read_ms = profile_event_elapsed_ms(read_event);
    clReleaseEvent(read_event);

    double total_ms = host_time_ms() - total_start_ms;

    /* ------------------------------------------------------------------
     * Validate device output against reference (exact match)
     * ------------------------------------------------------------------ */
    int valid = 0;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int h = 0; h < HEAD_NUM; h++) {
            for (int q = 0; q < Q_LEN; q++) {
                for (int k = 0; k < K_LEN; k++) {
                    if (output[b][h][q][k] != dbg_output[k]) {
//#if DEBUG_CODE
                        //printf(
                           // "dbg_output[%d] = %+1.16f is not equal to "
                            //"output[%d][%d][%d][%d] = %+1.16f\n",
                            //k, dbg_output[k], b, h, q, k, output[b][h][q][k]);
//#endif
                    } else {
                        valid++;
                    }
                }
            }
        }
    }

    printf("Valid: %d\n", valid);
    if (valid == BATCH_SIZE * K_LEN * HEAD_NUM * Q_LEN)
        printf("Softmax successful.\n");
    else
        printf("Softmax unsuccessful!\n");

    /* ------------------------------------------------------------------
     * Print and persist timing profile
     * ------------------------------------------------------------------ */
    printf("\n===== TIMING LOGS: Softmax =====\n");
    printf("Write time (score upload) : %.6f ms\n", write_ms);
    printf("Kernel execution time     : %.6f ms\n", kernel_ms);
    printf("Read time (output)        : %.6f ms\n", read_ms);
    printf("Profiled OpenCL time      : %.6f ms\n",
           write_ms + kernel_ms + read_ms);
    printf("Total host wall time      : %.6f ms\n", total_ms);
    printf("===============================\n");

    write_timing_profile("Softmax", write_ms, kernel_ms, read_ms, total_ms);

    /* ------------------------------------------------------------------
     * Release all OpenCL objects
     * ------------------------------------------------------------------ */
    clReleaseMemObject(dev_score);
    clReleaseMemObject(dev_output);

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

    return (valid == K_LEN) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/* =============================================================================
 * main()
 * =============================================================================*/
int main() {
  printf("#####################################################\n");
  printf("######  TESTING SOFTMAX (FPGA / OpenCL) ############\n");
  printf("#####################################################\n");

  // Load scores.
  float* dbg_score =
      load_vector_f32("debug_dumps/dbg_pre_softmax_l1_p7.bin", K_LEN);
  if(dbg_score) {
    printf("Loaded debug scores successfully\n");
  } else {
    printf("Did not load debug scores successfully\n");
    free(dbg_score);
    return 1;
  }
  // Load outputs.
  float* dbg_output =
      load_vector_f32("debug_dumps/dbg_post_softmax_l1_p7.bin", K_LEN);
  if(dbg_output) {
    printf("Loaded debug outputs successfully\n");
  } else {
    printf("Did not load debug outputs successfully\n");
    free(dbg_score);
    free(dbg_output);
    return 1;
  }

  // Initialise input score matrix.
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int h = 0; h < HEAD_NUM; h++) {
      for(int q = 0; q < Q_LEN; q++) {
        for(int k = 0; k < K_LEN; k++) {
          score[b][h][q][k] = dbg_score[k];
//#if DEBUG_CODE
          //if(k % 100 == 0)
           // printf("score[%d][%d][%d][%d] = %f\n", b, h, q, k,
                  // score[b][h][q][k]);
//#endif
        }
      }
    }
  }

  /* Run on FPGA via OpenCL. */
  int rc = run_softmax(dbg_output);

  free(dbg_score);
  free(dbg_output);

  return rc;
}