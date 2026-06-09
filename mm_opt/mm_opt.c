
#define __NUMCR__ (__NUMCOL__ * __NUMROW__)

#define M 4
#define N 4
#define K 4

#include "redefine.h"


#define KC 40
#define MC 130
#define NC 16
#define MR 4
#define NR 4


extern float *temp1;
extern float *temp2;
extern float *temp3;
extern float *Atilde;
extern float *Btilde;
extern int *data;


__hyperOp__ void hyOpLoopThree(__Op32 m$, __Op32 n$, __Op32 k$, __Op32 indexA$, __Op32 ldA$, __Op32 indexB$, __Op32 ldB$, __Op32 indexC$, __Op32 ldC$,__Op32 index1$,__Op32 index2$, __Op32 globalSync$);
__SMD__ smdHyOpLoopThree = {.ann = ANN_NONE, .arity = 12, .fptr = (__HyOpFunc)hyOpLoopThree};

__hyperOp__ void hyOpLoopTwo(__Op32 m$, __Op32 n$, __Op32 k$, __Op32 indexA$, __Op32 ldA$, __Op32 indexB$, __Op32 ldB$, __Op32 indexC$, __Op32 ldC$, __Op32 globalSync$);
__SMD__ smdHyOpLoopTwo = {.ann = ANN_NONE, .arity = 10, .fptr = (__HyOpFunc)hyOpLoopTwo};


__hyperOp__ void matmul_rank1(__Op32 m$, __Op32 n$, __Op32 k$, __Op32 A$, __Op32 ldA$, __Op32 B$, __Op32 ldB$, __Op32 C$, __Op32 ldC$, __Op32 localSync$)
{

  /*for(int p=0;p<k;p++){
    for(int j=0;j<n;j++){
      for(int i=0;i<m;i++)
      C[i*ldC+j]+=(A[i+p*m]*B[p*n+j]);
    }
  }*/




  float temp;
  int m= m$.i32;
  int n= n$.i32;
  int k= k$.i32;
  int p1;
  float *A = (float *)A$.ptr;
  float *B = (float *)B$.ptr;
  float *C = (float *)C$.ptr;
  int ldC=ldC$.i32;
  int ldA=ldA$.i32;
  int ldB=ldB$.i32;
  //int A1[4],B1[4],C1[16];

  float c00,c01,c02,c03,c10,c11,c12,c13,c20,c21,c22,c23,c30,c31,c32,c33;
  float a0,a1,a2,a3;
  float b0,b1,b2,b3;
  int i,j,m1=m,n1=n,x=0,y=0;
  float *c_row,*c_row1,*c_row2,*c_row3,*a_col,*b_row;
  //for(int x=0;x<m;x+=MR){


      // if(m>MR){
      //   if(x==0)
      //   m1=MR;
      //   else
      //   m1=m-MR;
      // }
      // else{
      //   m1=m;
      // }
      // if(n>NR){
      //   if(y==0)
      //   n1=NR;
      //   else
      //   n1=n-NR;
      // }
      // else{
      //   n1=n;
      // }

      if(m1==MR){
        if(n1==NR){

          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);
          //j=2+y;
          c02=*(c_row+2);
          //j=3+y;
          c03=*(c_row+3);

          c_row1=c_row+ldC;
          c10=*(c_row1);
          //j=1+y;
          c11=*(c_row1+1);
          //j=2+y;
          c12=*(c_row1+2);
          //j=3+y;
          c13=*(c_row1+3);

          c_row2=c_row1+ldC;
          c20=*(c_row2);
          //j=1+y;
          c21=*(c_row2+1);
          //j=2+y;
          c22=*(c_row2+2);
          //j=3+y;
          c23=*(c_row2+3);

          c_row3=c_row2+ldC;
          c30=*(c_row3);
          //j=1+y;
          c31=*(c_row3+1);
          //j=2+y;
          c32=*(c_row3+2);
          //j=3+y;
          c33=*(c_row3+3);
        }
        else if(n1==3){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);
          //j=2+y;
          c02=*(c_row+2);

          c_row1=c_row+ldC;
          c10=*(c_row1);
          //j=1+y;
          c11=*(c_row1+1);
          //j=2+y;
          c12=*(c_row1+2);

          c_row2=c_row1+ldC;
          c20=*(c_row2);
          //j=1+y;
          c21=*(c_row2+1);
          //j=2+y;
          c22=*(c_row2+2);

          c_row3=c_row2+ldC;
          c30=*(c_row3);
          //j=1+y;
          c31=*(c_row3+1);
          //j=2+y;
          c32=*(c_row3+2);
        }
        else if(n1==2){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);

          c_row1=c_row+ldC;
          c10=*(c_row1);
          //j=1+y;
          c11=*(c_row1+1);

          c_row2=c_row1+ldC;
          c20=*(c_row2);
          //j=1+y;
          c21=*(c_row2+1);

          c_row3=c_row2+ldC;
          c30=*(c_row3);
          //j=1+y;
          c31=*(c_row3+1);
        }
        else{
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);


          c_row1=c_row+ldC;
          c10=*(c_row1);


          c_row2=c_row1+ldC;
          c20=*(c_row2);


          c_row3=c_row2+ldC;
          c30=*(c_row3);

        }
      }
      else if(m1==3){
        if(n1==NR){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);
          //j=2+y;
          c02=*(c_row+2);
          //j=3+y;
          c03=*(c_row+3);

          c_row1=c_row+ldC;
          c10=*(c_row1);
          //j=1+y;
          c11=*(c_row1+1);
          //j=2+y;
          c12=*(c_row1+2);
          //j=3+y;
          c13=*(c_row1+3);

          c_row2=c_row1+ldC;
          c20=*(c_row2);
          //j=1+y;
          c21=*(c_row2+1);
          //j=2+y;
          c22=*(c_row2+2);
          //j=3+y;
          c23=*(c_row2+3);

        }
        else if(n1==3){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);
          //j=2+y;
          c02=*(c_row+2);

          c_row1=c_row+ldC;
          c10=*(c_row1);
          //j=1+y;
          c11=*(c_row1+1);
          //j=2+y;
          c12=*(c_row1+2);

          c_row2=c_row1+ldC;
          c20=*(c_row2);
          //j=1+y;
          c21=*(c_row2+1);
          //j=2+y;
          c22=*(c_row2+2);

        }
        else if(n1==2){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);

          c_row1=c_row+ldC;
          c10=*(c_row1);
          //j=1+y;
          c11=*(c_row1+1);

          c_row2=c_row1+ldC;
          c20=*(c_row2);
          //j=1+y;
          c21=*(c_row2+1);

        }
        else{
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);

          c_row1=c_row+ldC;
          c10=*(c_row1);

          c_row2=c_row1+ldC;
          c20=*(c_row2);

        }
      }
      else if(m1==2){
        if(n1==NR){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);
          //j=2+y;
          c02=*(c_row+2);
          //j=3+y;
          c03=*(c_row+3);

          c_row1=c_row+ldC;
          c10=*(c_row1);
          //j=1+y;
          c11=*(c_row1+1);
          //j=2+y;
          c12=*(c_row1+2);
          //j=3+y;
          c13=*(c_row1+3);

        }
        else if(n1==3){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);
          //j=2+y;
          c02=*(c_row+2);

          c_row1=c_row+ldC;
          c10=*(c_row1);
          //j=1+y;
          c11=*(c_row1+1);
          //j=2+y;
          c12=*(c_row1+2);

        }
        else if(n1==2){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);

          c_row1=c_row+ldC;
          c10=*(c_row1);
          //j=1+y;
          c11=*(c_row1+1);

        }
        else{
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);

          c_row1=c_row+ldC;
          c10=*(c_row1);

        }
      }
      else{
        if(n1==NR){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);
          //j=2+y;
          c02=*(c_row+2);
          //j=3+y;
          c03=*(c_row+3);

        }
        else if(n1==3){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);
          //j=2+y;
          c02=*(c_row+2);

        }
        else if(n1==2){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);
          //j=1+y;
          c01=*(c_row+1);

        }
        else{
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          c00=*(c_row);

        }
      }
      for(int p=0;p<k;p++){



        if(m1==MR){
          i=0;
          a_col=A+x*k+i+p*m1;
          a0=*(a_col);
          a1=*(a_col+1);
          a2=*(a_col+2);
          a3=*(a_col+3);
        }
        else if(m1==3){
          i=0;
          a_col=A+x*k+i+p*m1;
          a0=*(a_col);
          a1=*(a_col+1);
          a2=*(a_col+2);
        }
        else if(m1==2){
          i=0;
          a_col=A+x*k+i+p*m1;
          a0=*(a_col);
          a1=*(a_col+1);
        }
        else{
          i=0;
          a_col=A+x*k+i+p*m1;
          a0=*(a_col);
        }
        if(n1==NR){
          j=0;
          b_row=B+y*k+j+p*n1;
          b0=*(b_row);
          b1=*(b_row+1);
          b2=*(b_row+2);
          b3=*(b_row+3);
        }
        else if(n1==3){
          j=0;
          b_row=B+y*k+j+p*n1;
          b0=*(b_row);
          b1=*(b_row+1);
          b2=*(b_row+2);
        }
        else if(n1==2){
          j=0;
          b_row=B+y*k+j+p*n1;
          b0=*(b_row);
          b1=*(b_row+1);
        }
        else{
          j=0;
          b_row=B+y*k+j+p*n1;
          b0=*(b_row);
        }

        if(n1==NR){
          if(m1==MR){

            c00+=a0*b0;
            c10+=a1*b0;
            c20+=a2*b0;
            c30+=a3*b0;

            c01+=a0*b1;
            c11+=a1*b1;
            c21+=a2*b1;
            c31+=a3*b1;

            c02+=a0*b2;
            c12+=a1*b2;
            c22+=a2*b2;
            c32+=a3*b2;

            c03+=a0*b3;
            c13+=a1*b3;
            c23+=a2*b3;
            c33+=a3*b3;

          }
          else if(m1==3){

            c00+=a0*b0;
            c10+=a1*b0;
            c20+=a2*b0;

            c01+=a0*b1;
            c11+=a1*b1;
            c21+=a2*b1;

            c02+=a0*b2;
            c12+=a1*b2;
            c22+=a2*b2;

            c03+=a0*b3;
            c13+=a1*b3;
            c23+=a2*b3;
          }
          else if(m1==2){

            c00+=a0*b0;
            c10+=a1*b0;

            c01+=a0*b1;
            c11+=a1*b1;

            c02+=a0*b2;
            c12+=a1*b2;

            c03+=a0*b3;
            c13+=a1*b3;
          }
          else{

            c00+=a0*b0;

            c01+=a0*b1;

            c02+=a0*b2;

            c03+=a0*b3;
          }
        }
        else if(n1==3){
          if(m1==MR){

            c00+=a0*b0;
            c10+=a1*b0;
            c20+=a2*b0;
            c30+=a3*b0;

            c01+=a0*b1;
            c11+=a1*b1;
            c21+=a2*b1;
            c31+=a3*b1;

            c02+=a0*b2;
            c12+=a1*b2;
            c22+=a2*b2;
            c32+=a3*b2;
          }
          else if(m1==3){

            c00+=a0*b0;
            c10+=a1*b0;
            c20+=a2*b0;

            c01+=a0*b1;
            c11+=a1*b1;
            c21+=a2*b1;

            c02+=a0*b2;
            c12+=a1*b2;
            c22+=a2*b2;
          }
          else if(m1==2){

            c00+=a0*b0;
            c10+=a1*b0;

            c01+=a0*b1;
            c11+=a1*b1;

            c02+=a0*b2;
            c12+=a1*b2;
          }
          else{

            c00+=a0*b0;

            c01+=a0*b1;

            c02+=a0*b2;
          }
        }
        else if(n1==2){
          if(m1==MR){

            c00+=a0*b0;
            c10+=a1*b0;
            c20+=a2*b0;
            c30+=a3*b0;

            c01+=a0*b1;
            c11+=a1*b1;
            c21+=a2*b1;
            c31+=a3*b1;
          }
          else if(m1==3){

            c00+=a0*b0;
            c10+=a1*b0;
            c20+=a2*b0;

            c01+=a0*b1;
            c11+=a1*b1;
            c21+=a2*b1;
          }
          else if(m1==2){

            c00+=a0*b0;
            c10+=a1*b0;

            c01+=a0*b1;
            c11+=a1*b1;
          }
          else{

            c00+=a0*b0;

            c01+=a0*b1;
          }
        }
        else{
          if(m1==MR){

            c00+=a0*b0;
            c10+=a1*b0;
            c20+=a2*b0;
            c30+=a3*b0;
          }
          else if(m1==3){

            c00+=a0*b0;
            c10+=a1*b0;
            c20+=a2*b0;
          }
          else if(m1==2){

            c00+=a0*b0;
            c10+=a1*b0;
          }
          else{

            c00+=a0*b0;
          }
        }

      }

      if(m1==MR){
        if(n1==NR){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;
          //j=2+y;
          *(c_row+2)=c02;
          //j=3+y;
          *(c_row+3)=c03;

          c_row1=c_row+ldC;
          *(c_row1)=c10;
          //j=1+y;
          *(c_row1+1)=c11;
          //j=2+y;
          *(c_row1+2)=c12;
          //j=3+y;
          *(c_row1+3)=c13;

          c_row2=c_row1+ldC;
          *(c_row2)=c20;
          //j=1+y;
          *(c_row2+1)=c21;
          //j=2+y;
          *(c_row2+2)=c22;
          //j=3+y;
          *(c_row2+3)=c23;

          c_row3=c_row2+ldC;
          *(c_row3)=c30;
          //j=1+y;
          *(c_row3+1)=c31;
          //j=2+y;
          *(c_row3+2)=c32;
          //j=3+y;
          *(c_row3+3)=c33;
        }
        else if(n1==3){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;
          //j=2+y;
          *(c_row+2)=c02;

          c_row1=c_row+ldC;
          *(c_row1)=c10;
          //j=1+y;
          *(c_row1+1)=c11;
          //j=2+y;
          *(c_row1+2)=c12;

          c_row2=c_row1+ldC;
          *(c_row2)=c20;
          //j=1+y;
          *(c_row2+1)=c21;
          //j=2+y;
          *(c_row2+2)=c22;

          c_row3=c_row2+ldC;
          *(c_row3)=c30;
          //j=1+y;
          *(c_row3+1)=c31;
          //j=2+y;
          *(c_row3+2)=c32;
        }
        else if(n1==2){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;

          c_row1=c_row+ldC;
          *(c_row1)=c10;
          //j=1+y;
          *(c_row1+1)=c11;

          c_row2=c_row1+ldC;
          *(c_row2)=c20;
          //j=1+y;
          *(c_row2+1)=c21;

          c_row3=c_row2+ldC;
          *(c_row3)=c30;
          //j=1+y;
          *(c_row3+1)=c31;

        }
        else{
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;

          c_row1=c_row+ldC;
          *(c_row1)=c10;

          c_row2=c_row1+ldC;
          *(c_row2)=c20;

          c_row3=c_row2+ldC;
          *(c_row3)=c30;
        }
      }
      else if(m1==3){
        if(n1==NR){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;
          //j=2+y;
          *(c_row+2)=c02;
          //j=3+y;
          *(c_row+3)=c03;

          c_row1=c_row+ldC;
          *(c_row1)=c10;
          //j=1+y;
          *(c_row1+1)=c11;
          //j=2+y;
          *(c_row1+2)=c12;
          //j=3+y;
          *(c_row1+3)=c13;

          c_row2=c_row1+ldC;
          *(c_row2)=c20;
          //j=1+y;
          *(c_row2+1)=c21;
          //j=2+y;
          *(c_row2+2)=c22;
          //j=3+y;
          *(c_row2+3)=c23;

        }
        else if(n1==3){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;
          //j=2+y;
          *(c_row+2)=c02;

          c_row1=c_row+ldC;
          *(c_row1)=c10;
          //j=1+y;
          *(c_row1+1)=c11;
          //j=2+y;
          *(c_row1+2)=c12;

          c_row2=c_row1+ldC;
          *(c_row2)=c20;
          //j=1+y;
          *(c_row2+1)=c21;
          //j=2+y;
          *(c_row2+2)=c22;

        }
        else if(n1==2){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;

          c_row1=c_row+ldC;
          *(c_row1)=c10;
          //j=1+y;
          *(c_row1+1)=c11;

          c_row2=c_row1+ldC;
          *(c_row2)=c20;
          //j=1+y;
          *(c_row2+1)=c21;

        }
        else{
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;

          c_row1=c_row+ldC;
          *(c_row1)=c10;

          c_row2=c_row1+ldC;
          *(c_row2)=c20;

        }
      }
      else if(m1==2){
        if(n1==NR){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;
          //j=2+y;
          *(c_row+2)=c02;
          //j=3+y;
          *(c_row+3)=c03;

          c_row1=c_row+ldC;
          *(c_row1)=c10;
          //j=1+y;
          *(c_row1+1)=c11;
          //j=2+y;
          *(c_row1+2)=c12;
          //j=3+y;
          *(c_row1+3)=c13;

        }
        else if(n1==3){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;
          //j=2+y;
          *(c_row+2)=c02;

          c_row1=c_row+ldC;
          *(c_row1)=c10;
          //j=1+y;
          *(c_row1+1)=c11;
          //j=2+y;
          *(c_row1+2)=c12;

        }
        else if(n1==2){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;

          c_row1=c_row+ldC;
          *(c_row1)=c10;
          //j=1+y;
          *(c_row1+1)=c11;

        }
        else{
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;

          c_row1=c_row+ldC;
          *(c_row1)=c10;

        }
      }
      else{
        if(n1==NR){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;
          //j=2+y;
          *(c_row+2)=c02;
          //j=3+y;
          *(c_row+3)=c03;

        }
        else if(n1==3){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;
          //j=2+y;
          *(c_row+2)=c02;

        }
        else if(n1==2){
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;
          //j=1+y;
          *(c_row+1)=c01;

        }
        else{
          i=x;
          j=y;
          c_row=C+i*ldC+j;
          *(c_row)=c00;

        }
      }







  __CMAddr syncAddr = localSync$.cmAddr;
  //synchronization with localjoin
  __sync(syncAddr, -1);
}
__SMD__ smd_matmul_rank1 = {.ann = ANN_NONE, .arity = 10, .fptr = (__HyOpFunc)matmul_rank1};

__hyperOp__ void end(){
  re_sigEndOfKernel(0);
}
__SMD__ smdEnd = {.ann =ANN_JOIN, .arity = 1, .fptr = (__HyOpFunc)end};

__hyperOp__ void hyOpLoopOneSync(__Op32 globalSync$){



  __CMAddr syncAddr = globalSync$.cmAddr;
  //synchronization with globaljoin
  __sync(syncAddr, -1);
}
__SMD__ smdHyOpLoopOneSync = {.ann = ANN_NONE|ANN_JOIN, .arity = 2, .fptr = (__HyOpFunc)hyOpLoopOneSync};


__hyperOp__ void hyOpLoopOne(__Op32 m$, __Op32 n$, __Op32 k$, __Op32 A$, __Op32 ldA$,__Op32 B$, __Op32 ldB$, __Op32 C$, __Op32 ldC$, __Op32 globalSync$){

  __CMAddr localSync = globalSync$.cmAddr;


  int n= n$.i32;
  int m= m$.i32;
  int k= k$.i32;


  float *A = (float *)A$.ptr;
  float *B = (float *)B$.ptr;
  float *C = (float *)C$.ptr;
  int ldC=ldC$.i32;
  int ldA=ldA$.i32;
  int ldB=ldB$.i32;

  for ( int j=0; j<n; j+=NR ){
    int jb;
    if(NR <= (n-j))
    jb=NR;
    else
    jb=(n-j);
    __CMAddr frId = __createInst(&smd_matmul_rank1);
    __writeCM( frId, m);
    __writeCM( re_opAddr(frId,1), jb);
    __writeCM( re_opAddr(frId,2), k);
    __writeCM( re_opAddr(frId,3), A);
    __writeCM( re_opAddr(frId,4), ldA);
    __writeCM( re_opAddr(frId,5), &B[j*k]);
    __writeCM( re_opAddr(frId,6), ldB);
    __writeCM( re_opAddr(frId,7), &C[(j)]);
    __writeCM( re_opAddr(frId,8), ldC);
    __writeCM( re_opAddr(frId, 9), localSync);
  }

}
__SMD__ smdHyOpLoopOne = {.ann = ANN_NONE, .arity = 10, .fptr = (__HyOpFunc)hyOpLoopOne};

__hyperOp__ void hyOpLoopTwoSync( __CMAddr selfId, __Op32 globalSync$){
  __CMAddr syncAddr = globalSync$.cmAddr;
  //synchronization with globaljoin
  __sync(syncAddr, -1);
}
__SMD__ smdHyOpLoopTwoSync = {.ann = ANN_NONE|ANN_JOIN, .arity = 2, .fptr = (__HyOpFunc)hyOpLoopTwoSync};


__hyperOp__ void hyOpLoopTwo(__Op32 m$, __Op32 n$, __Op32 k$, __Op32 indexA$, __Op32 ldA$, __Op32 indexB$, __Op32 ldB$, __Op32 indexC$, __Op32 ldC$, __Op32 globalSync$){




  __CMAddr localSync = globalSync$.cmAddr;


  int n= n$.i32;
  int m= m$.i32;
  int k= k$.i32;




  int indexA = indexA$.i32;
  int indexB = indexB$.i32;
  int indexC = indexC$.i32;
  int ldA=ldA$.i32;
  int ldB=ldB$.i32;
  int ldC=ldC$.i32;


  for ( int i=0; i<m; i+=(MR) ){
    int ib;
    if((MR) <= (m-i))
    ib=(MR);
    else
    ib=(m-i);
    __CMAddr frId = __createInst(&smdHyOpLoopOne);
    __writeCM( frId, ib);
    __writeCM( re_opAddr(frId,1), n);
    __writeCM( re_opAddr(frId,2), k);
    __writeCM( re_opAddr(frId,3), &Atilde[indexA+i*k]);
    __writeCM( re_opAddr(frId,4), ldA);
    __writeCM( re_opAddr(frId,5), &Btilde[indexB]);
    __writeCM( re_opAddr(frId,6), ldB);
    __writeCM( re_opAddr(frId,7), &temp3[indexC+(i)*ldC]);
    __writeCM( re_opAddr(frId,8), ldC);
    __writeCM( re_opAddr(frId, 9), localSync);
  }


}


__hyperOp__ void PackMicroPanelA_MRxKC(__Op32 m$, __Op32 k$, __Op32 A$, __Op32 ldA$, __Op32 Atilde$, __Op32 localSync$ )
/* Pack a micro-panel of A into buffer pointed to by Atilde. */
{

  int m= m$.i32;
  int k= k$.i32;
  int ldA= ldA$.i32;
  float *A = (float *)A$.ptr;
  float *Atilde = (float *)Atilde$.ptr;
    for ( int p=0; p<k; p++ ){
      for ( int i=0; i<m; i++ )
	*Atilde++ = A[i*ldA+p];
    }
    __CMAddr syncAddr = localSync$.cmAddr;
    //synchronization with localjoin
    __sync(syncAddr, -1);
}
__SMD__ smdPackMicroPanelA_MRxKC = {.ann = ANN_NONE, .arity = 6, .fptr = (__HyOpFunc)PackMicroPanelA_MRxKC};



__hyperOp__ void PackBlockA_MxKCSync(__Op32 m$,__Op32 n$, __Op32 k$,__Op32 indexA$, __Op32 ldA$,__Op32 indexB$, __Op32 ldB$,__Op32 indexC1$, __Op32 ldC$,__Op32 index1$,__Op32 index2$, __Op32 globalSync$)
{
  int m= m$.i32;
  int n= n$.i32;
  int k= k$.i32;
  int indexA= indexA$.i32;
  int ldA= ldA$.i32;
  int indexB= indexB$.i32;
  int ldB= ldB$.i32;
  int indexC1= indexC1$.i32;
  int ldC= ldC$.i32;
  int index1= index1$.i32;
  int index2= index2$.i32;
  __CMAddr localSync = globalSync$.cmAddr;
  // __CMAddr frId = __createInst(&smdHyOpLoopTwo);
  // __writeCM( frId, m);
  // __writeCM( re_opAddr(frId,1), n);
  // __writeCM( re_opAddr(frId,2), k);
  // __writeCM( re_opAddr(frId,3), indexA);
  // __writeCM( re_opAddr(frId,4), ldA);
  // __writeCM( re_opAddr(frId,5), indexB);
  // __writeCM( re_opAddr(frId,6), ldB);
  // __writeCM( re_opAddr(frId,7), indexC1);
  // __writeCM( re_opAddr(frId,8), ldC);
  // __writeCM( re_opAddr(frId, 9), localSync);


  __CMAddr frId = __createInst(&smdHyOpLoopThree);
  __writeCM( frId, m);
  __writeCM( re_opAddr(frId,1), n);
  __writeCM( re_opAddr(frId,2), k);
  __writeCM( re_opAddr(frId,3), indexA);
  __writeCM( re_opAddr(frId,4), ldA);
  __writeCM( re_opAddr(frId,5), indexB);
  __writeCM( re_opAddr(frId,6), ldB);
  __writeCM( re_opAddr(frId,7), indexC1);
  __writeCM( re_opAddr(frId,8), ldC);
  __writeCM( re_opAddr(frId,9), index1);
  __writeCM( re_opAddr(frId,10), index2);
  __writeCM( re_opAddr(frId, 11), localSync);

}
__SMD__ smdPackBlockA_MxKCSync = {.ann = ANN_NONE|ANN_JOIN, .arity = 13, .fptr = (__HyOpFunc)PackBlockA_MxKCSync};


void PackBlockA_MxKC( int m, int n, int k, int indexA, int ldA,int indexB, int ldB,int indexC1, int ldC, int indexAtilde,int index1,int index2, __CMAddr localSync1)
/* Pack a MC x KC block of A.  MC is assumed to be a multiple of MR.  The block is
   packed into Atilde a micro-panel at a time. */
{
  __CMAddr PackBlockA_MxKCSyncId = __createInst(&smdPackBlockA_MxKCSync);
  __CMAddr localSync = re_opAddr(PackBlockA_MxKCSyncId,15);

  __writeCM( PackBlockA_MxKCSyncId, m);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,1), n);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,2), k);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,3), indexAtilde);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,4), ldA);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,5), indexB);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,6), ldB);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,7), indexC1);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,8), ldC);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,9), index1);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId,10), index2);
  __writeCM( re_opAddr(PackBlockA_MxKCSyncId, 11), localSync1);

  int nHyOps;
  if(m%MR==0)
  nHyOps=m/MR;
  else
  nHyOps=m/MR+1;
  __sync( localSync, nHyOps);
  float *Atilde1;
  Atilde1=&Atilde[indexAtilde];
  for ( int i=0; i<m; i+= MR ){
    int ib;
    if(MR<=(m-i))
    ib=MR;
    else
    ib=(m-i);
    __CMAddr frId = __createInst(&smdPackMicroPanelA_MRxKC);
    __writeCM( frId, ib);
    __writeCM( re_opAddr(frId,1), k);
    __writeCM( re_opAddr(frId,2), &temp1[indexA+i*ldA]);
    __writeCM( re_opAddr(frId,3), ldA);
    __writeCM( re_opAddr(frId,4), Atilde1);
    __writeCM( re_opAddr(frId, 5), localSync);
    Atilde1 += ib * k;
  }
}

__hyperOp__ void PackMicroPanelB_KCxNR( __Op32 k$, __Op32 n$, __Op32 B$, __Op32 ldB$, __Op32 Btilde$, __Op32 localSync$ )
/* Pack a micro-panel of B into buffer pointed to by Btilde. */
{

  int k= k$.i32;
  int n= n$.i32;
  int ldB= ldB$.i32;
  float *B = (float *)B$.ptr;
  float *Btilde = (float *)Btilde$.ptr;

    for ( int p=0; p<k; p++ ){
      for ( int j=0; j<n; j++ )
	     *Btilde++ = B[p*ldB+j];
    }




    __CMAddr syncAddr = localSync$.cmAddr;
    //synchronization with localjoin
    __sync(syncAddr, -1);
}
__SMD__ smdPackMicroPanelB_KCxNR = {.ann = ANN_NONE, .arity = 6, .fptr = (__HyOpFunc)PackMicroPanelB_KCxNR};


__hyperOp__ void PackPanelB_KCxNJCSync(__Op32 m$, __Op32 n$, __Op32 k$, __Op32 indexA$, __Op32 ldA$, __Op32 indexBtilde$, __Op32 ldB$, __Op32 indexC$, __Op32 ldC$, __Op32 globalSync$ )
{
  int m= m$.i32;
  int n= n$.i32;
  int k= k$.i32;
  int indexA= indexA$.i32;
  int ldA= ldA$.i32;
  int indexBtilde= indexBtilde$.i32;
  int ldB= ldB$.i32;
  int indexC= indexC$.i32;
  int ldC= ldC$.i32;
  __CMAddr localSync = globalSync$.cmAddr;
  // __CMAddr frId = __createInst(&smdHyOpLoopThree);
  // __writeCM( frId, m);
  // __writeCM( re_opAddr(frId,1), n);
  // __writeCM( re_opAddr(frId,2), k);
  // __writeCM( re_opAddr(frId,3), indexA);
  // __writeCM( re_opAddr(frId,4), ldA);
  // __writeCM( re_opAddr(frId,5), indexBtilde);
  // __writeCM( re_opAddr(frId,6), ldB);
  // __writeCM( re_opAddr(frId,7), indexC);
  // __writeCM( re_opAddr(frId,8), ldC);
  // __writeCM( re_opAddr(frId,9), index1);
  // __writeCM( re_opAddr(frId,10), index2);
  // __writeCM( re_opAddr(frId, 11), localSync);

  for ( int i=0; i<m; i+=(MR) ){
    int ib;
    if((MR) <= (m-i))
    ib=(MR);
    else
    ib=(m-i);
    __CMAddr frId = __createInst(&smdHyOpLoopOne);
    __writeCM( frId, ib);
    __writeCM( re_opAddr(frId,1), n);
    __writeCM( re_opAddr(frId,2), k);
    __writeCM( re_opAddr(frId,3), &Atilde[indexA+i*k]);
    __writeCM( re_opAddr(frId,4), ldA);
    __writeCM( re_opAddr(frId,5), &Btilde[indexBtilde]);
    __writeCM( re_opAddr(frId,6), ldB);
    __writeCM( re_opAddr(frId,7), &temp3[indexC+(i)*ldC]);
    __writeCM( re_opAddr(frId,8), ldC);
    __writeCM( re_opAddr(frId, 9), localSync);
  }
}
__SMD__ smdPackPanelB_KCxNJCSync = {.ann = ANN_NONE|ANN_JOIN, .arity = 11, .fptr = (__HyOpFunc)PackPanelB_KCxNJCSync};

void PackPanelB_KCxNJC( int m, int n, int k,int indexA, int ldA,int indexB, int ldB, int indexC, int ldC, int indexBtilde, __CMAddr localSync1 )
/* Pack a KC x NC panel of B.  NC is assumed to be a multiple of NR.  The block is
   packed into Btilde a micro-panel at a time. */
{
  __CMAddr PackPanelB_KCxNJCSyncId = __createInst(&smdPackPanelB_KCxNJCSync);
  __writeCM( PackPanelB_KCxNJCSyncId, m);
  __writeCM( re_opAddr(PackPanelB_KCxNJCSyncId,1), n);
  __writeCM( re_opAddr(PackPanelB_KCxNJCSyncId,2), k);
  __writeCM( re_opAddr(PackPanelB_KCxNJCSyncId,3), indexA);
  __writeCM( re_opAddr(PackPanelB_KCxNJCSyncId,4), ldA);
  __writeCM( re_opAddr(PackPanelB_KCxNJCSyncId,5), indexBtilde);
  __writeCM( re_opAddr(PackPanelB_KCxNJCSyncId,6), ldB);
  __writeCM( re_opAddr(PackPanelB_KCxNJCSyncId,7), indexC);
  __writeCM( re_opAddr(PackPanelB_KCxNJCSyncId,8), ldC);
  __writeCM( re_opAddr(PackPanelB_KCxNJCSyncId, 9), localSync1);
  __CMAddr localSync = re_opAddr(PackPanelB_KCxNJCSyncId,15);
  int nHyOps;
  if(n%NR==0)
  nHyOps=n/NR;
  else
  nHyOps=n/NR+1;
  __sync( localSync, nHyOps);
  float *Btilde1;
  Btilde1=&Btilde[indexBtilde];
  for ( int j=0; j<n; j+= NR ){
    int jb;
    if(NR<=(n-j))
    jb=NR;
    else
    jb=(n-j);
    __CMAddr frId = __createInst(&smdPackMicroPanelB_KCxNR);
    __writeCM( frId, k);
    __writeCM( re_opAddr(frId,1), jb);
    //__writeCM( re_opAddr(frId,2), pb);
    __writeCM( re_opAddr(frId,2), &temp2[indexB+j]);
    __writeCM( re_opAddr(frId,3), ldB);
    __writeCM( re_opAddr(frId,4), Btilde1);
    __writeCM( re_opAddr(frId, 5), localSync);
    //PackMicroPanelB_KCxNR( k, jb, &B[j], ldB, Btilde );
    Btilde1 +=  k*jb;
  }
}

__hyperOp__ void hyOpLoopThreeSync(__Op32 globalSync$, __Op32 index1$,__Op32 index2$){

  int pb,m,n,k,p,indexA,indexB,indexC,ldA,ldB,ldC,y;

  int index1 = index1$.i32;
  int index2 = index2$.i32;
  __CMAddr syncAddr = globalSync$.cmAddr;
  //synchronization with globaljoin
  __sync(syncAddr, -1);
  int d0,d1;
  d0=__NUMROW__;
  d1=__NUMCOL__;
  if((M/d0+(d0-1))%MC==0)
  y=(M/d0+(d0-1))/MC;
  else
  y=(M/d0+(d0-1))/MC+1;
  if((*(data+index1*11*y+index2*11+4))==1)
  {
    m=*((data+index1*11*y+index2*11+0));
    n=*(data+index1*11*y+index2*11+1);
    k=*(data+index1*11*y+index2*11+2);
    p=*(data+index1*11*y+index2*11+3);

    indexA=*(data+index1*11*y+index2*11+5);
    indexB=*(data+index1*11*y+index2*11+6);
    indexC=*(data+index1*11*y+index2*11+7);
    ldA=*(data+index1*11*y+index2*11+8);
    ldB=*(data+index1*11*y+index2*11+9);
    ldC=*(data+index1*11*y+index2*11+10);


    p+=KC;
    if(KC<=(k-p))
    pb=KC;
    else
    pb=(k-p);

    *(data+index1*11*y+index2*11+3)=p;
    if((p+KC)>=k)
    *(data+index1*11*y+index2*11+4)=0;
    indexA=indexA+p;
    indexB=indexB+p*ldB;

    // PackPanelB_KCxNC( m,n,pb,indexA, ldA,indexB, ldB,indexC,ldC,index1,index2, index1*KC*NC*y+index2*KC*NC,syncAddr );
    PackBlockA_MxKC( m,n,pb,indexA, ldA,indexB, ldB,indexC,ldC, index1*MC*KC*y+index2*KC*MC,index1,index2,syncAddr );


  }

}
__SMD__ smdHyOpLoopThreeSync = {.ann = ANN_NONE|ANN_JOIN, .arity = 4, .fptr = (__HyOpFunc)hyOpLoopThreeSync};

__hyperOp__ void hyOpLoopThree( __Op32 m$, __Op32 n$, __Op32 k$, __Op32 indexA$, __Op32 ldA$, __Op32 indexB$, __Op32 ldB$, __Op32 indexC$, __Op32 ldC$,__Op32 index1$,__Op32 index2$, __Op32 globalSync$){


  __CMAddr hyOpLoopThreeSyncId = __createInst(&smdHyOpLoopThreeSync);
  __writeCM(hyOpLoopThreeSyncId, globalSync$);
  __CMAddr localSync = re_opAddr(hyOpLoopThreeSyncId,15);



  int indexA = indexA$.i32;
  int indexB = indexB$.i32;
  int indexC = indexC$.i32;
  int index1 = index1$.i32;
  int index2 = index2$.i32;
  __writeCM(re_opAddr(hyOpLoopThreeSyncId,1), index1);
  __writeCM(re_opAddr(hyOpLoopThreeSyncId,2), index2);
  int ldA=ldA$.i32;
  int ldB=ldB$.i32;
  int ldC=ldC$.i32;
  int n= n$.i32;
  int m= m$.i32;
  int k= k$.i32;
  int p;
  int nHyOps,nHyOps1,x,y=0,val1;
  int d0,d1;
  d0=__NUMROW__;
  d1=__NUMCOL__;

  if(m%MR == 0)
  x=m/MR;
  else
  x=m/MR+1;
  nHyOps=x;
  int jb;

  for ( int j=0; j<n; j+=NC ){
    if(NC <= (n-j))
    jb=NC;
    else
    jb=(n-j);
    if(jb%(NR) == 0)
    y+=(jb/(NR));
    else
    y+=(jb/(NR)+1);
  }
  nHyOps=nHyOps*y;
  __sync( localSync, nHyOps);
  if((M/d0+(d0-1))%MC==0)
  val1=(M/d0+(d0-1))/MC;
  else
  val1=(M/d0+(d0-1))/MC+1;
  for ( int j=0; j<n; j+=NC ){

    if(NC <= (n-j))
    jb=NC;
    else
    jb=(n-j);
    PackPanelB_KCxNJC( m,jb, k, indexA, ldA,indexB+j,ldB,indexC+j, ldC,index1*KC*(N/d1+(d1-1))*val1+index2*KC*(N/d1+(d1-1))+KC*j,localSync );

  }

}

__hyperOp__ void hyOpLoopFourSync(__Op32 globalSync$){

  __CMAddr syncAddr = globalSync$.cmAddr;
  // //synchronization with globaljoin
  __sync(syncAddr, -1);
}
__SMD__ smdHyOpLoopFourSync = {.ann = ANN_NONE|ANN_JOIN, .arity = 2, .fptr = (__HyOpFunc)hyOpLoopFourSync};

__hyperOp__ void hyOpLoopFour(__Op32 m$, __Op32 n$, __Op32 k$, __Op32 indexA$, __Op32 ldA$, __Op32 indexB$, __Op32 ldB$, __Op32 indexC$, __Op32 ldC$,__Op32 index1$, __Op32 index2$,__Op32 addr$,  __Op32 globalSync$){

  __CMAddr hyOpLoopFourSyncId = __createInst(&smdHyOpLoopFourSync);
  __writeCM(hyOpLoopFourSyncId, globalSync$);
  __CMAddr localSync = re_opAddr(hyOpLoopFourSyncId,15);
int *addr=addr$.ptr;
//*addr=0;
int x1=*addr;
printf("Pratik:%d\n",x1);
int next=0,p=0,pb,x,y;
int n= n$.i32;
int m= m$.i32;
int k= k$.i32;
int indexA = indexA$.i32;
int indexB = indexB$.i32;
int indexC = indexC$.i32;
int index1 = index1$.i32;
int index2 = index2$.i32;
int ldA=ldA$.i32;
int ldB=ldB$.i32;
int ldC=ldC$.i32;
int nHyOps;
int d0,d1;
d0=__NUMROW__;
d1=__NUMCOL__;
if(k%KC==0)
nHyOps=k/KC;
else
nHyOps=k/KC+1;
__sync( localSync, nHyOps);
if((M/d0+(d0-1))%MC==0)
y=(M/d0+(d0-1))/MC;
else
y=(M/d0+(d0-1))/MC+1;
*(data+index1*11*y+index2*11+0)=m;
*(data+index1*11*y+index2*11+1)=n;
*(data+index1*11*y+index2*11+2)=k;
*(data+index1*11*y+index2*11+3)=p;
*(data+index1*11*y+index2*11+4)=next;
*(data+index1*11*y+index2*11+5)=indexA;
*(data+index1*11*y+index2*11+6)=indexB;
*(data+index1*11*y+index2*11+7)=indexC;
*(data+index1*11*y+index2*11+8)=ldA;
*(data+index1*11*y+index2*11+9)=ldB;
*(data+index1*11*y+index2*11+10)=ldC;

    if(KC <= (k-p))
    pb=KC;
    else
    pb=(k-p);
    if((p+KC)<k)
    *(data+index1*11*y+index2*11+4)=1;

    PackBlockA_MxKC( m,n,pb,indexA, ldA,indexB, ldB,indexC,ldC, index1*MC*KC*y+index2*MC*KC,index1,index2,localSync );


}
__SMD__ smdHyOpLoopFour = {.ann = ANN_NONE, .arity = 13, .fptr = (__HyOpFunc)hyOpLoopFour};

__hyperOp__ void hyOpLoopFiveSync( __Op32 globalSync$){

  __CMAddr syncAddr = globalSync$.cmAddr;
  // //synchronization with globaljoin
  __sync(syncAddr, -1);
}
__SMD__ smdHyOpLoopFiveSync = {.ann = ANN_NONE|ANN_JOIN, .arity = 2, .fptr = (__HyOpFunc)hyOpLoopFiveSync};


__hyperOp__ void hyOpLoopFive( __Op32 m$, __Op32 n$, __Op32 k$, __Op32 indexA$, __Op32 ldA$, __Op32 indexB$, __Op32 ldB$, __Op32 indexC$, __Op32 ldC$, __Op32 index1$,__Op32 globalSync$){

  int *addr;
  __CMAddr hyOpLoopFiveSyncId = __createInst(&smdHyOpLoopFiveSync);
  __writeCM(hyOpLoopFiveSyncId, globalSync$);
  __CMAddr localSync = re_opAddr(hyOpLoopFiveSyncId,15);


  int x=9;
  addr=&x;
  int n= n$.i32;
  int m= m$.i32;
  int k= k$.i32;
  int nHyOps;
  if(m%MC == 0){
    nHyOps=m/MC;
  }
  else
  nHyOps=m/MC+1;

  int indexA = indexA$.i32;
  int indexB = indexB$.i32;
  int indexC = indexC$.i32;
  int index1 = index1$.i32;
  int ldA=ldA$.i32;
  int ldB=ldB$.i32;
  int ldC=ldC$.i32;

  __sync( localSync, nHyOps);


  for ( int i=0; i<m; i+=MC ){
    int ib;
    if(MC <= (m-i))
    ib=MC;
    else
    ib=(m-i);
    __CMAddr frId = __createInst(&smdHyOpLoopFour);
    __writeCM( frId, ib);
    __writeCM( re_opAddr(frId,1), n);
    __writeCM( re_opAddr(frId,2), k);
    __writeCM( re_opAddr(frId,3), indexA+i*ldA);
    __writeCM( re_opAddr(frId,4), ldA);
    __writeCM( re_opAddr(frId,5), indexB);
    __writeCM( re_opAddr(frId,6), ldB);
    __writeCM( re_opAddr(frId,7), indexC+i*ldC);
    __writeCM( re_opAddr(frId,8), ldC);
    __writeCM( re_opAddr(frId,9), index1);
    __writeCM( re_opAddr(frId,10), i/MC);
    __writeCM( re_opAddr(frId, 11), addr);
    __writeCM( re_opAddr(frId, 12), localSync);
  }
  __fDelete(re_getSelfID());
}
__SMD__ smdHyOpLoopFive = {.ann = ANN_NONE, .arity = 11, .fptr = (__HyOpFunc)hyOpLoopFive};


__kernel void __REDEFINE_main()
{
  int val1;
  int d0,d1,x,y,iA,iB,i_s0;
  d0=__NUMROW__;
  d1=__NUMCOL__;
  /*for(int i=0;i<(M/d0+(d0-1))*d0;i++){
    for(int j=0;j<(N/d1+(d1-1))*d1;j++)
    C_elem_dist[i*(N/d1+(d1-1))*d1+j]=0;
  }*/



  int nHyOps=__NUMCR__;
  int val,indexA,indexB=0,indexC,indexM;

  //fork-join parallelism across CRs.
  __CMAddr endId = __createInst(&smdEnd);
  __CMAddr globalSync = re_opAddr(endId, 15);
  __sync( globalSync, nHyOps );

  for(int j=0; j< __NUMROW__ ; j++){
    for(int i=0; i< __NUMCOL__ ; i++){
      re_CrId crid = {.idX=i, .idY=j};
      __CMAddr frId = __rFAlloc(1, crid);
      __fBind(frId, &smdHyOpLoopFive);
      __writeCM( frId, (M/d0+(d0-1)));
      __writeCM( re_opAddr(frId,1), (N/d1+(d1-1)));
      __writeCM( re_opAddr(frId,2), K);
      __writeCM( re_opAddr(frId,3), j*(M/d0+(d0-1))*K);
      __writeCM( re_opAddr(frId,4), K);
      __writeCM( re_opAddr(frId,5), i*K*(N/d1+(d1-1)));
      __writeCM( re_opAddr(frId,6), (N/d1+(d1-1)));
      __writeCM( re_opAddr(frId,7), j*(M/d0+(d0-1))*(N/d1+(d1-1))*d1+i*(M/d0+(d0-1))*(N/d1+(d1-1)));
      __writeCM( re_opAddr(frId,8), (N/d1+(d1-1)));
      __writeCM( re_opAddr(frId,9), j*d1+i);
      __writeCM( re_opAddr(frId,10), globalSync);
    }
  }
}
