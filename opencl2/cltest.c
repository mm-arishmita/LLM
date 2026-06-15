#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 220
#include "CL/opencl.h"

#include "att.h"           /* HEAD_DIM, ATTN_HEADS, SEQ_LEN, CE_USE_NUM, mat_t */
#include "common.h"  /* create_device(), checkErr(), DeviceName           */


#define PROGRAM_FILE "att.c"

/* =============================================================================
 * TARGET DEVICE
 * =============================================================================*/
#if defined(__FUNCTSIM__)
DeviceName targetDevice = rfsim_hyperop;   /* functional simulator */
#else
DeviceName targetDevice = rsim_hyperop;    /* RTL simulator / FPGA */
#endif

/* =============================================================================
 * HOST-SIDE BUFFERS
 * =============================================================================*/
static mat_t h_Q      [ATTN_HEADS][HEAD_DIM];
static mat_t h_K      [SEQ_LEN   ][HEAD_DIM];
static mat_t h_Scores [ATTN_HEADS][SEQ_LEN ];
static mat_t h_manual [ATTN_HEADS][SEQ_LEN ];

/* =============================================================================
 * MANUAL REFERENCE
 * =============================================================================*/
static void manual_attention_scores(void)
{
    float inv_sqrt_dim = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            float acc = 0.0;
            for (int d = 0; d < HEAD_DIM; d++)
                acc += (float)h_Q[h][d] * (float)h_K[t][d];
            h_manual[h][t] = (float)(acc * (float)inv_sqrt_dim);
        }
    }
}

/* =============================================================================
 * run_attention()
 * =============================================================================*/
static int run_attention(void)
{
    cl_int error = 0;

    /* ------------------------------------------------------------------
     * OpenCL init
     * FSim:  redefine_initialize(1, 1)
     * FPGA:  create_device → clCreateContext → clCreateCommandQueueWithProperties
     * ------------------------------------------------------------------ */
    cl_device_id     deviceId  = create_device(targetDevice);

    cl_context       contextId = clCreateContext(NULL, 1, &deviceId,
                                                 NULL, NULL, &error);
    checkErr(error, "clCreateContext");

    cl_command_queue cmdqId    = clCreateCommandQueueWithProperties(
                                     contextId, deviceId, NULL, &error);
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

    /* "att_start"  ↔  __kernel void att_start(...) in att.c */
    cl_kernel kernelId = clCreateKernel(programId, "att_start", &error);
    checkErr(error, "clCreateKernel att_start");

    /* ------------------------------------------------------------------
     * Allocate device buffers
     *
     * FSim:  (none — Q, K, Scores are globals shared with the kernel)
     * FPGA:  clCreateBuffer for each tensor
     *
     *  dev_Q        READ_ONLY  | COPY_HOST_PTR  → upload h_Q at creation
     *  dev_K        READ_ONLY  | COPY_HOST_PTR  → upload h_K at creation
     *  dev_partials READ_WRITE                  → device-internal scratch
     *  dev_Scores   WRITE_ONLY                  → read back after kernel
     * ------------------------------------------------------------------ */
    size_t sz_Q        = sizeof(mat_t) * ATTN_HEADS * HEAD_DIM;
    size_t sz_K        = sizeof(mat_t) * SEQ_LEN    * HEAD_DIM;
    size_t sz_partials = sizeof(mat_t) * ATTN_HEADS * SEQ_LEN * CE_USE_NUM;
    size_t sz_Scores   = sizeof(mat_t) * ATTN_HEADS * SEQ_LEN;

    cl_mem dev_Q = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sz_Q, h_Q, &error);
    checkErr(error, "clCreateBuffer dev_Q");

    cl_mem dev_K = clCreateBuffer(contextId,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sz_K, h_K, &error);
    checkErr(error, "clCreateBuffer dev_K");

    cl_mem dev_partials = clCreateBuffer(contextId,
                                          CL_MEM_READ_WRITE,
                                          sz_partials, NULL, &error);
    checkErr(error, "clCreateBuffer dev_partials");

    cl_mem dev_Scores   = clCreateBuffer(contextId,
                                          CL_MEM_WRITE_ONLY,
                                          sz_Scores, NULL, &error);
    checkErr(error, "clCreateBuffer dev_Scores");

    /* ------------------------------------------------------------------
     * Set kernel arguments  →  att_start(Q, K, partials,scores)
     * ------------------------------------------------------------------ */
    error = clSetKernelArg(kernelId, 0, sizeof(cl_mem), &dev_Q);
    checkErr(error, "clSetKernelArg dev_Q");

    error = clSetKernelArg(kernelId, 1, sizeof(cl_mem), &dev_K);
    checkErr(error, "clSetKernelArg dev_K");

    error = clSetKernelArg(kernelId, 2, sizeof(cl_mem), &dev_partials);
    checkErr(error, "clSetKernelArg dev_partials");

    error = clSetKernelArg(kernelId, 3, sizeof(cl_mem), &dev_Scores);
    checkErr(error, "clSetKernelArg dev_Scores");

    /* ------------------------------------------------------------------
     * Launch kernel
     * ------------------------------------------------------------------ */
    size_t global[1] = {4};
    size_t local[1]  = {1};

    error = clEnqueueNDRangeKernel(cmdqId, kernelId, 1,
                                   NULL, global, local,
                                   0, NULL, NULL);
    checkErr(error, "clEnqueueNDRangeKernel");

    error = clFinish(cmdqId);
    checkErr(error, "clFinish");

    /* ------------------------------------------------------------------
     * Read Scores back to host
     * ------------------------------------------------------------------ */
    error = clEnqueueReadBuffer(cmdqId, dev_Scores,
                                CL_TRUE, 0, sz_Scores,
                                h_Scores,
                                0, NULL, NULL);
    checkErr(error, "clEnqueueReadBuffer dev_Scores");

    /* ------------------------------------------------------------------
     * Validate device output against reference
     * ------------------------------------------------------------------ */
    int valid = 1;
    for (int h = 0; h < ATTN_HEADS; h++) {
        for (int t = 0; t < SEQ_LEN; t++) {
            if (h_Scores[h][t] != h_manual[h][t]) {
                valid = 0;
                printf("[MISMATCH] Scores[%d][%d]: device=%f  ref=%f\n",
                       h, t, h_Scores[h][t], h_manual[h][t]);
            }
        }
    }

    if (valid)
        printf("Attention Score computation successful.\n");
    else
        printf("Attention Score computation unsuccessful!\n");

    /* ------------------------------------------------------------------
     * Release all OpenCL objects
     * ------------------------------------------------------------------ */
    clReleaseMemObject(dev_Q);
    clReleaseMemObject(dev_K);
    clReleaseMemObject(dev_partials);
    clReleaseMemObject(dev_Scores);

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

/* =============================================================================
 * main()
 * =============================================================================*/
int main(void)
{
    printf("#####################################################\n");
    printf("######  TESTING ATTENTION SCORE (FPGA / OpenCL) ####\n");
    printf("#####################################################\n");

    /* -----------------------------------------------------------------------
     * Initialise Q
     * ----------------------------------------------------------------------- */
    for (int h = 0; h < ATTN_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++)
            h_Q[h][d] = (float)((h + 1) * (d + 1)) / (float)HEAD_DIM;

    /* -----------------------------------------------------------------------
     * Initialise K
     * ----------------------------------------------------------------------- */
    for (int t = 0; t < SEQ_LEN; t++)
        for (int d = 0; d < HEAD_DIM; d++)
            h_K[t][d] = (t % 2 == 0)
                        ? ((d % 2 == 0) ? 1.0f : 0.0f)
                        : (float)(d + 1) / (float)HEAD_DIM;

    /* -----------------------------------------------------------------------
     * Compute manual reference
     * ----------------------------------------------------------------------- */
    manual_attention_scores();

    /* -----------------------------------------------------------------------
     * Run on FPGA via OpenCL
     * ----------------------------------------------------------------------- */
    return run_attention();
}
