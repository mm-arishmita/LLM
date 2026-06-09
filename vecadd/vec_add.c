#include "redefine.h"

float A[8];
float B[8];
float C[8];

__hyperOp__ void End();
__SMD__ smd_End = {.ann = ANN_JOIN, .arity = 1, .fptr = (__HyOpFunc)End};

__hyperOp__ void End() {
  re_println("Reached end hyperop !!\n");
  re_sigEndOfKernel(0);
}

__hyperOp__ void vecAdd(__Op32, __Op32, __Op32, __Op32);
__SMD__ smd_vecAdd = {.ann = ANN_NONE, .arity = 4, .fptr = (__HyOpFunc)vecAdd};


__hyperOp__ void vecAdd(__Op32 a, __Op32 b, __Op32 c,
                        __Op32 endFr) {

  float *A = a.ptr;
  float *B = b.ptr;
  float *C = c.ptr;

  //__CMAddr confrId = consumerFrId.cmAddr;

  int i;
  for (i = 0; i < 8; i++) {
    C[i] = A[i] + B[i];
  }

  //__writeCM(confrId, 0);
  __sync(endFr.cmAddr, -1);
}

__kernel void __REDEFINE_main() {
     __CMAddr endFr = __createInst(&smd_End);
     __sync(re_opAddr(endFr, 15), 1);

     __CMAddr addFr = __createInst(&smd_vecAdd);
    
    re_println("MatAddFrId:%d, endFrId:%d\n", addFr, endFr);

   __writeCM(re_opAddr(addFr, 0), (void *)(&A[0]));
   __writeCM(re_opAddr(addFr, 1), (void *)(&B[0]));
   __writeCM(re_opAddr(addFr, 2), (void *)(&C[0]));
   __writeCM(re_opAddr(addFr, 3), re_opAddr(endFr, 15));
}
