#include "redefine.h"


__hyperOp__ void End();
__SMD__ smd_End = {.ann = ANN_JOIN, .arity = 1, .fptr = (__HyOpFunc)End};
__hyperOp__ void End() {
  re_println("End HyperOp !!\n");
  re_sigEndOfKernel(0);
}

__hyperOp__ void SayHello(__Op32, __Op32, __Op32);
__SMD__ smd_SayHello = {.ann = ANN_NONE, .arity = 3, .fptr = (__HyOpFunc)SayHello};
__hyperOp__ void SayHello(__Op32 r0, __Op32 r1, __Op32 endFr) {
     int n = r0.i32;
     int m = r1.i32;
  re_println("Hello-World: %d", n);
  re_println("Hi i am: %d", m);
  __sync(endFr.cmAddr, -1);
}

__kernel void __REDEFINE_main() {

 // __CMAddr HelloFr = __createInst(&smd_SayHello);
  __CMAddr endFr = __createInst(&smd_End);
  int fabricSize = (__NUMROW__ * __NUMCOL__);
  __sync(re_opAddr(endFr, 15), fabricSize);

  for (int y = 0; y < __NUMROW__; y++) {
    for (int x = 0; x < __NUMCOL__; x++) {
      re_CrId crid = {.idX = x, .idY = y};
      __CMAddr frId = __rFAlloc(1, crid);
      __fBind(frId, &smd_SayHello);
     // __writeCM(frId, (y * __NUMCOL__) + x);
      __writeCM(re_opAddr(frId, 0), (y * __NUMCOL__) + x);
      __writeCM(re_opAddr(frId, 1),  x+y);
      __writeCM(re_opAddr(frId, 2), re_opAddr(endFr, 15));
    }
  }
    re_println("REDEFINE_main End");
   
}

