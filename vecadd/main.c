#include "redefine.h"
#include <stdio.h>

//inputs decalration
extern float A[256];
extern float B[256];
extern float C[256];
void send_input(){
    for (int i=0;i<8;i++){
        A[i] = (float)i;
        B[i] = (float)i;
    }
    printf("Elments of A:\n");
    for(int i =0; i<8; i++){
        printf("%4.2f ",A[i]);
    }
    printf("\n");
    printf("Elments of B:\n");
    for(int i =0; i<8; i++){
        printf("%4.2f ",B[i]);
    }
    printf("\n");
}
void receive_output(){
    printf("Elments of C:\n");
    for(int i =0; i<8; i++){
        printf("%4.2f ",C[i]);
    }
    printf("\n");
}
int main(){
    send_input();

    redefine_initialize(1,1);
    __re_StartHyperOpInit(0, 0, 1, 1);
    printf("start hyperop\n");
    __REDEFINE_main(); //execute start hyperop
    printf("excuted start hyperop\n");
    redefine_execute(); //execute ready heperops
    printf("Executed kernel code");

    receive_output();
}