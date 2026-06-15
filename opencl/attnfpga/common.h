
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

typedef enum DeviceName {
  rsim,
  rsim_hyperop,
  rfsim,
  rfsim_hyperop,
  any
} DeviceName;

cl_device_id create_device(DeviceName name);

void checkErr(cl_int err, const char *name);

void checkKernelEnqueue(cl_int err);

void showDeviceInfo(cl_device_id device_id);
