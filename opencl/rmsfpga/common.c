#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "common.h"

// bool startsWith(const char *pre, const char *str)
// {
// 	size_t lenpre = strlen(pre),
// 	       lenstr = strlen(str);
// 	return lenstr < lenpre ? false : memcmp(pre, str, lenpre) == 0;
// }

cl_device_id create_device(DeviceName devName) {

  cl_device_id device_id = NULL;
  int err;
  cl_platform_id platform_id[3];
  cl_uint ret_num_platforms;

  cl_device_id *devices;
  cl_uint ret_num_devices;

  char *pocl_platform_name = "Portable Computing Language";

  char device_name[32];
  switch (devName) {
  case rsim:
    // OpenCL-C ISA simulator
    strcpy(device_name, "rsim");
    break;
  case rsim_hyperop:
    // C-with-hyperOps ISA simulator
    strcpy(device_name, "rsim-hyperop");
    break;
  case rfsim:
    // OpenCL-C functional simulator
    strcpy(device_name, "rfsim");
    break;
  case rfsim_hyperop:
    // C-with-hyperops functional simulator
    strcpy(device_name, "rfsim-hyperop");
    break;
  default:
    strcpy(device_name, "any");
    break;
  }

  /* Identify a platform */
  /* Get Platform and Device Info */
  err = clGetPlatformIDs(3, platform_id, &ret_num_platforms);
  printf("No. of platforms detected: %d\n", ret_num_platforms);

  cl_platform_id pocl_platform_id;
  for (int i = 0; i < ret_num_platforms; i++) {
    size_t namelen;
    err = clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 1024, NULL,
                            &namelen);
    char *pname = calloc(namelen, sizeof(char));
    err = clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, namelen, pname,
                            NULL);
    if (strcmp(pname, pocl_platform_name) == 0) {
      printf("Selecting platform : %s\n", pname);
      pocl_platform_id = platform_id[i];
    }
  }

  /* Access a device */
  err = clGetDeviceIDs(pocl_platform_id, CL_DEVICE_TYPE_ALL, 6, NULL,
                       &ret_num_devices);

  devices = (cl_device_id *)malloc(sizeof(cl_device_id) * ret_num_devices);
  err = clGetDeviceIDs(pocl_platform_id, CL_DEVICE_TYPE_ALL, ret_num_devices,
                       devices, NULL);
  if (err < 0) {
    fprintf(stderr, "Couldn't find device");
    exit(EXIT_FAILURE);
  }
  printf("Devices found : %d \n", ret_num_devices);

  if (devName != any) {
    for (cl_uint i = 0; i < ret_num_devices; i++) {
      char name_data[128];
      err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name_data),
                            name_data, NULL);
      if (err < 0) {
        fprintf(stderr, "Couldn't find device info ");
        exit(EXIT_FAILURE);
      }
      printf("Device %s\n", name_data);
      if (strcmp(device_name, name_data) == 0) {
        printf("Selecting : %s\n", name_data);
        device_id = devices[i];
        break;
      }
    }
    if(device_id == NULL){
	printf("No matching device found,defaulting to first device");
	device_id = devices[0];
	printf("Using device index 0 as fallback");
   }

  } else {
    char name_data[128];
    err = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(name_data),
                          name_data, NULL);
    if (err < 0) {
      fprintf(stderr, "Couldn't find device info ");
      exit(EXIT_FAILURE);
    }
    printf("Device %s\n", name_data);
    device_id = devices[0];
  }

  return device_id;
}

void checkErr(cl_int err, const char *name) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: %s\n", name);
    exit(EXIT_FAILURE);
  }
}

void checkKernelEnqueue(cl_int err) {
  if (err != CL_SUCCESS) {
    switch (err) {
    case CL_INVALID_PROGRAM_EXECUTABLE:
      fprintf(stderr, "Invalid program executable\n");
      break;
    case CL_INVALID_COMMAND_QUEUE:
      fprintf(stderr, "Invalid command queue\n");
      break;
    case CL_INVALID_KERNEL:
      fprintf(stderr, "Invalid kernel\n");
      break;
    case CL_INVALID_CONTEXT:
      fprintf(stderr, "Invalid context\n");
      break;
    case CL_INVALID_KERNEL_ARGS:
      fprintf(stderr, "Invalid kernel args\n");
      break;
    case CL_INVALID_WORK_DIMENSION:
      fprintf(stderr, "Invalid work dimension\n");
      break;
    case CL_INVALID_WORK_GROUP_SIZE:
      fprintf(stderr, "Invalid work group size\n");
      break;
    case CL_OUT_OF_HOST_MEMORY:
      fprintf(stderr, "out of host memory\n");
      break;
    default:
      fprintf(stderr, "UNKNOWN\n");
      break;
    }
  }
}

void showDeviceInfo(cl_device_id device_id) {
  cl_int ret;
  char *value;
  size_t valuesize;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valuesize);
  checkErr(ret, "DEVICE NAME");
  value = (char *)malloc(valuesize);
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, valuesize, value, NULL);
  printf("Device: %s\n", value);
  free(value);

  ret = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, 0, NULL, &valuesize);
  checkErr(ret, "DEVICE VERSION");
  value = (char *)malloc(valuesize);
  clGetDeviceInfo(device_id, CL_DEVICE_VERSION, valuesize, value, NULL);
  printf("Hardware version: %s\n", value);
  free(value);

  ret = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, 0, NULL, &valuesize);
  checkErr(ret, "DRIVER VERSION");
  value = (char *)malloc(valuesize);
  clGetDeviceInfo(device_id, CL_DRIVER_VERSION, valuesize, value, NULL);
  printf("Driver version: %s\n", value);
  free(value);

  ret = clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, 0, NULL,
                        &valuesize);
  checkErr(ret, "OpenCL C version");
  value = (char *)malloc(valuesize);
  clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, valuesize, value,
                  NULL);
  printf("OpenCL C Version: %s\n", value);
  free(value);

  cl_uint maxComputeUnits;
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
  checkErr(ret, "Compute Units");
  printf("Compute units: %d\n", maxComputeUnits);
}
