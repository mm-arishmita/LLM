#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>

#define CL_TARGET_OPENCL_VERSION 220
#include "CL/opencl.h"

#include "common_defs.h"
#include "common.h"

#define PROGRAM_FILE "rms.c"

/* =============================================================================
 * TARGET DEVICE
 * =============================================================================*/
#if defined(__FUNCTSIM__)
DeviceName targetDevice = rfsim_hyperop;   /* functional simulator */
#else
DeviceName targetDevice = rsim_hyperop;    /* RTL simulator / FPGA */
#endif

/* HOST BUFFER DECLARATIONS */

static mat_t input      [BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE];
static mat_t hidden     [BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE];
static mat_t output     [BATCH_SIZE][SEQ_LEN][HIDDEN_SIZE];
static mat_t weights    [HIDDEN_SIZE];

/* HOST DEFINITIONS */

//#ifndef DEBUG_CODE
//#define DEBUG_CODE 0
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
    fprintf(timing_log, "Write time (input,hidden,weights upload) : %.6f ms\n", write_ms);
    fprintf(timing_log, "Kernel execution time   : %.6f ms\n", kernel_ms);
    fprintf(timing_log, "Read time (output)      : %.6f ms\n", read_ms);
    fprintf(timing_log, "Profiled OpenCL time    : %.6f ms\n",
            write_ms + kernel_ms + read_ms);
    fprintf(timing_log, "Total host wall time    : %.6f ms\n", total_ms);
    fprintf(timing_log, "===============================\n");
    fclose(timing_log);
    printf("Timing profile written to timing_profile.txt\n");
}

/* HOST CODE */

float* load_vector_f32(const char *path, int expected_size) {
  FILE *f = fopen(path, "rb");
  if(!f) {
    perror("fopen failed");
    return NULL;
  }

  float *data = (float*)malloc(sizeof(float) * expected_size);
  if(!data) {
    fclose(f);
    return NULL;
  }

  size_t read = fread(data, sizeof(float), expected_size, f);
  fclose(f);

  if(read != (size_t)expected_size) {
    fprintf(stderr, "Size mismatch: expected %d, got %zu\n",
            expected_size, read);
    free(data);
    return NULL;
  }

  return data;
}

void build_fname(char *out, size_t sz, const char *dir, const char *tag,
                 int layer, int pos) {
  snprintf(out, sz, "%s/dbg_%s_l%d_p%d.bin", dir, tag, layer, pos);
}

LayerDebug load_layer_debug(const char *dir, int layer, int pos, int hidden)
{
    LayerDebug d = {0};
    char fname[256];

    // ---- RMS 1 ----
    build_fname(fname, sizeof(fname), dir, "rms1", layer, pos);
    d.rms1_in = load_vector_f32(fname, hidden);

    build_fname(fname, sizeof(fname), dir, "rms1_out", layer, pos);
    d.rms1_out = load_vector_f32(fname, hidden);

    // ---- Residual 1 ----
    build_fname(fname, sizeof(fname), dir, "res1", layer, pos);
    d.res1_in = load_vector_f32(fname, hidden);

    build_fname(fname, sizeof(fname), dir, "res1_out", layer, pos);
    d.res1_out = load_vector_f32(fname, hidden);

    // ---- Attention output ----
    build_fname(fname, sizeof(fname), dir, "attn_out", layer, pos);
    d.attn_out = load_vector_f32(fname, hidden);

    // ---- RMS 2 ----
    build_fname(fname, sizeof(fname), dir, "rms2", layer, pos);
    d.rms2_in = load_vector_f32(fname, hidden);

    build_fname(fname, sizeof(fname), dir, "rms2_out", layer, pos);
    d.rms2_out = load_vector_f32(fname, hidden);

    // ---- Residual 2 ----
    build_fname(fname, sizeof(fname), dir, "res2", layer, pos);
    d.res2_in = load_vector_f32(fname, hidden);

    build_fname(fname, sizeof(fname), dir, "res2_out", layer, pos);
    d.res2_out = load_vector_f32(fname, hidden);

    // ---- MLP output ----
    build_fname(fname, sizeof(fname), dir, "mlp_out", layer, pos);
    d.mlp_out = load_vector_f32(fname, hidden);

    return d;
}

void free_layer_debug(LayerDebug *d) {
  free(d->rms1_in);
  free(d->rms1_out);
  free(d->res1_in);
  free(d->res1_out);
  free(d->rms2_in);
  free(d->rms2_out);
  free(d->res2_in);
  free(d->res2_out);
  free(d->attn_out);
  free(d->mlp_out);
}

/* =============================================================================
 * run_rmsnorm()
 * =============================================================================*/
static int run_rmsnorm(LayerDebug *dbg)
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

    /* "rms_start"  ↔  __kernel void rms_start(...) in rms.c */
    cl_kernel kernelId = clCreateKernel(programId, "rms_start", &error);
    checkErr(error, "clCreateKernel rms_start");

    /* ------------------------------------------------------------------
     * Allocate device buffers
     *
     *  dev_input    READ_ONLY   → uploaded via write event
     *  dev_hidden   READ_ONLY   → uploaded via write event
     *  dev_weights  READ_ONLY   → uploaded via write event
     *  dev_output   WRITE_ONLY  → read back after kernel
     * ------------------------------------------------------------------ */
    size_t sz_input   = sizeof(mat_t) * BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE;
    size_t sz_hidden  = sizeof(mat_t) * BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE;
    size_t sz_weights = sizeof(mat_t) * HIDDEN_SIZE;
    size_t sz_output  = sizeof(mat_t) * BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE;

    cl_mem dev_input = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY,
                                  sz_input, NULL, &error);
    checkErr(error, "clCreateBuffer dev_input");

    cl_mem dev_hidden = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY,
                                  sz_hidden, NULL, &error);
    checkErr(error, "clCreateBuffer dev_hidden");

    cl_mem dev_weights = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY,
                                  sz_weights, NULL, &error);
    checkErr(error, "clCreateBuffer dev_weights");

    cl_mem dev_output = clCreateBuffer(contextId,
                                  CL_MEM_WRITE_ONLY,
                                  sz_output, NULL, &error);
    checkErr(error, "clCreateBuffer dev_output");

    /* ------------------------------------------------------------------
     * Upload input, hidden, weights (timed via profiling events)
     * ------------------------------------------------------------------ */
    cl_event write_input_event   = NULL;
    cl_event write_hidden_event  = NULL;
    cl_event write_weights_event = NULL;

    error = clEnqueueWriteBuffer(cmdqId, dev_input, CL_FALSE, 0,
                                 sz_input, input, 0, NULL, &write_input_event);
    checkErr(error, "clEnqueueWriteBuffer dev_input");

    error = clEnqueueWriteBuffer(cmdqId, dev_hidden, CL_FALSE, 0,
                                 sz_hidden, hidden, 0, NULL, &write_hidden_event);
    checkErr(error, "clEnqueueWriteBuffer dev_hidden");

    error = clEnqueueWriteBuffer(cmdqId, dev_weights, CL_FALSE, 0,
                                 sz_weights, weights, 0, NULL, &write_weights_event);
    checkErr(error, "clEnqueueWriteBuffer dev_weights");

    error = clFinish(cmdqId);
    checkErr(error, "clFinish (after writes)");

    double write_ms = profile_event_elapsed_ms(write_input_event)
                     + profile_event_elapsed_ms(write_hidden_event)
                     + profile_event_elapsed_ms(write_weights_event);

    clReleaseEvent(write_input_event);
    clReleaseEvent(write_hidden_event);
    clReleaseEvent(write_weights_event);

    /* ------------------------------------------------------------------
     * Set kernel arguments  →  rms_start(input, hidden, weights, output)
     * ------------------------------------------------------------------ */
    error = clSetKernelArg(kernelId, 0, sizeof(cl_mem), &dev_input);
    checkErr(error, "clSetKernelArg dev_input");

    error = clSetKernelArg(kernelId, 1, sizeof(cl_mem), &dev_hidden);
    checkErr(error, "clSetKernelArg dev_hidden");

    error = clSetKernelArg(kernelId, 2, sizeof(cl_mem), &dev_weights);
    checkErr(error, "clSetKernelArg dev_weights");

    error = clSetKernelArg(kernelId, 3, sizeof(cl_mem), &dev_output);
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
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                if (output[b][s][i] != dbg->rms2_out[i]) {
//#if DEBUG_CODE
                    //printf("dbg.rms2_out[%d][%d][%d] = %+2.5f is not equal to output[%d][%d][%d] = %+2.5f\n",
                           b, s, i, dbg->rms2_out[i], b, s, i, output[b][s][i]);
//#endif
                } else {
                    valid++;
                }
            }
        }
    }

    printf("Valid: %d\n", valid);
    if (valid == HIDDEN_SIZE)
        printf("RMSNorm successful.\n");
    else
        printf("RMSNorm unsuccessful!\n");

    /* ------------------------------------------------------------------
     * Print and persist timing profile
     * ------------------------------------------------------------------ */
    printf("\n===== TIMING LOGS: RMSNorm =====\n");
    printf("Write time (input,hidden,weights upload) : %.6f ms\n", write_ms);
    printf("Kernel execution time   : %.6f ms\n", kernel_ms);
    printf("Read time (output)      : %.6f ms\n", read_ms);
    printf("Profiled OpenCL time    : %.6f ms\n",
           write_ms + kernel_ms + read_ms);
    printf("Total host wall time    : %.6f ms\n", total_ms);
    printf("===============================\n");

    write_timing_profile("RMSNorm", write_ms, kernel_ms, read_ms, total_ms);

    /* ------------------------------------------------------------------
     * Release all OpenCL objects
     * ------------------------------------------------------------------ */
    clReleaseMemObject(dev_input);
    clReleaseMemObject(dev_hidden);
    clReleaseMemObject(dev_weights);
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

    return (valid == HIDDEN_SIZE) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/// Main function.
int main(void) {

    printf("#####################################################\n");
    printf("######  TESTING RMSNORM (FPGA / OpenCL) ############\n");
    printf("#####################################################\n");

  LayerDebug dbg = load_layer_debug("debug_dumps", 1, 0, HIDDEN_SIZE);
  if (dbg.rms1_in && dbg.rms1_out) {
    printf("Loaded debug tensors successfully\n");
  }
  else {
    printf("Did not load debug tensors successfully\n");
    free_layer_debug(&dbg);
    return 1;
  }

  // Initialise input and hidden matrix.
  for(int b = 0; b < BATCH_SIZE; b++) {
    for(int s = 0; s < SEQ_LEN; s++) {
      for(int i = 0; i < HIDDEN_SIZE; i++) {
        input[b][s][i] = dbg.rms1_in[i];
//#if DEBUG_CODE
        //printf("input[%d][%d][%d] = %f\n", b, s, i, input[b][s][i]);
//#endif
        hidden[b][s][i] = dbg.attn_out[i];
//#if DEBUG_CODE
        //printf("hidden[%d][%d][%d] = %f\n", b, s, i, hidden[b][s][i]);
//#endif
      }
    }
  }

  // Load weights.
  float *rms_weights = load_vector_f32("debug_dumps/rmsnorm_layer1.bin", HIDDEN_SIZE * 2);
  if (rms_weights) {
    printf("Loaded rms weights successfully\n");
  }
  else {
    printf("Did not load rms weights successfully\n");
    free_layer_debug(&dbg);
    free(rms_weights);
    return 1;
  }

  for(int i = 0; i < HIDDEN_SIZE; i++) {
    // Add HIDDEN_SIZE to read post_attention_layernorm weights.
    weights[i] = rms_weights[HIDDEN_SIZE + i];
  }

     /* Run on FPGA via OpenCL. */
    int rc = run_rmsnorm(&dbg);

    free_layer_debug(&dbg);
    free(rms_weights);

    return rc;
}
