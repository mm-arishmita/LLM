#include<stdio.h>
#include "redefine.h"

void __REDEFINE_main();

int main(){
    redefine_initialize(__NUMROW__, __NUMCOL__);
    __re_StartHyperOpInit(0, 0, __NUMCOL__, __NUMROW__);
    __REDEFINE_main();
    redefine_execute();
}