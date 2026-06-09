
/* Include files */
#include "redefine.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


//#include "redefine.h"


#define dabs( x ) ( (x) < 0 ? -(x) : x )
#define __NUMCOL__ 4
#define __NUMROW__ 4
#define KC 40
  #define MC 130
  #define NC 16
  #define MR 4
  #define NR 4
const int M = 4;
const int N = 4;
const int K = 4;
float *A;
	float *B;
	float *C;
  float *Atilde;
	float *Btilde;
	float *Cref;
	float *temp1;
	float *temp2;
  float *temp3;
	int *data;


float MaxAbsDiff( int m, int n, float *ap, int lda, float *bp, int ldb )
/*
   MaxAbsDiff returns the maximum absolute difference over
   corresponding elements of matrices A and B.
*/
{
  float diff=0.0;
  int  i, j;

  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
      if ( dabs( ap[i*lda+j] - bp[i*ldb+j] ) > diff )
	  diff = dabs( ap[i*lda+j] - bp[i*ldb+j] );

  return diff;
}


void RandomMatrix( int m, int n, float *ap, int lda )
/*
   RandomMatrix overwrite A with random values.
*/
{
  int  i, j;

	srand(time(NULL));
  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
      ap[i*lda+j] = (rand() % 10);
}



void initMatrix(float *a, int n, int m){
  int i = 0;
  int j = 0;
  for(i = 0; i < n; i = i+1){
		for(j = 0; j < m; j = j+1){
			a[i*m+j] = 10 * ((float)rand()/RAND_MAX);
    }
  }
}

void zeroInitMatrix(float *a, int n, int m){
	int i = 0;
	int j = 0;
  for(i = 0; i < n; i = i+1){
		for(j = 0; j < m; j = j+1){
			a[i*m+j] = (float)0;
		}
	}
}

void showMatrix(float *a, int m, int n){
  for (int i = 0; i < m; i++) {
 		for (int j = 0; j < n; j++) {
 			printf("%f ", a[i*n+j]);
 		}
 		printf("\n");
 	}
}

void MyGemm( int m, int n, int k, float *A, int ldA,
	     float *B, int ldB, float *C, int ldC )
{
  for ( int i=0; i<m; i++ )
    for ( int j=0; j<n; j++ )
      for ( int p=0; p<k; p++ )
        C[ (i)*ldC + j ] += A[ (i)*ldA + p ] * B[ (p)*ldB + j ];
}

int runmatmul()
{

  
	int szA, szB, szC,val1,val2,iNode[__NUMROW__][__NUMCOL__],jNode[__NUMROW__][__NUMCOL__],x,y;
  int rows = 0;
	int columns,n,m,k,ldA,ldB,ldC;
	float n1,diff,d_one = 1.0;
	int count = 0;
  char ch;
	n=N;
	m=M;
	k=K;
	ldA=K;
	ldB=N;
	ldC=N;
//  #define MPC 4
  szA  = M*K;
	szB  = K*N;
	szC  = M*N;
	A = (float *)malloc(szA*sizeof(float));
	B = (float *)malloc(szB*sizeof(float));
	C = (float *)malloc(szC*sizeof(float));
	Cref=(float *)malloc(szC*sizeof(float));
	printf("Reached Here 00\n");
	//RandomMatrix( m, k, A, ldA );
	//showMatrix(A, m,k);
	//RandomMatrix( k, n, B, ldB );
	//showMatrix(B, k,n);
        printf("Enter elements of A (%dx%d):\n", m, k);
  	for (int i = 0; i < m; i++) {
    		for (int j = 0; j < k; j++) {
       			 scanf("%f", &A[i*ldA + j]);
    		}
	}

 	 printf("Enter elements of B (%dx%d):\n", k, n);
  	 for (int i = 0; i < k; i++) {
   		 for (int j = 0; j < n; j++) {
       			 scanf("%f", &B[i*ldB + j]);
    		}
	}

      printf("Reached Here 000\n");
	MyGemm(m,n,k,A,ldA,B,ldB,Cref,ldC);
  printf("Reached Here 001\n");
  int d0,d1;
  d0=__NUMROW__;
  d1=__NUMCOL__;
	if((M/d0+(d0-1))%MC==0)
	val1=(M/d0+(d0-1))/MC;
	else
	val1=(M/d0+(d0-1))/MC+1;
   printf("Reached Here 03\n");
	temp1 = malloc((M/d0+(d0-1))*K*d0*sizeof(float));
  printf("Reached Here 04\n");
	temp2 = (float *)malloc(K*(N/d1+(d1-1))*d1*sizeof(float));
  printf("Reached Here 05\n");
  temp3 = (float *)malloc((M/d0+(d0-1))*(N/d1+(d1-1))*d0*d1*sizeof(float));
  printf("Reached Here 02\n");
	data = (int *)malloc(val1*11*d0*d1*sizeof(int));
  Atilde = (float *)malloc(MC*KC*val1*d0*d1*sizeof(float));
  Btilde = (float *)malloc(KC*(N/d1+(d1-1))*val1*d0*d1*sizeof(float));
  printf("Reached Here 01\n");
  for(int i=0;i<(M/d0+(d0-1))*d0;i++){
    for(int j=0;j<(N/d1+(d1-1))*d1;j++)
    temp3[i*(N/d1+(d1-1))*d1+j]=0;
  }
  for(int i=0;i<__NUMROW__;i++){
    for(int j=0;j<__NUMCOL__;j++){
      iNode[i][j]=0;
      jNode[i][j]=0;
    }
  }
  for(int i=0;i<M;i++){
    x=(i%d0);

    for(int j=0;j<K;j++){
        temp1[x*(M/d0+(d0-1))*K+iNode[x][0]*K+jNode[x][0]]=A[i*K+j];
        jNode[x][0]++;

    }

    for(int p=0;p<__NUMROW__;p++){
      for(int q=0;q<__NUMCOL__;q++){
        if(p==x)
        iNode[p][q]++;
        jNode[p][q]=0;
      }
    }

  }

  for(int i=0;i<__NUMROW__;i++){
    for(int j=0;j<__NUMCOL__;j++){
      iNode[i][j]=0;
      jNode[i][j]=0;
    }
  }

  /*for(int i=0;i<K;i++){
    for(int j=0;j<(N/d1+(d1-1))*d1;j++){
      temp2[i*(N/d1+(d1-1))*d1+j]=-1;
    }
  }*/

  for(int j=0;j<N;j++){
    y=(j%d1);


    for(int i=0;i<K;i++){

        temp2[y*(N/d1+(d1-1))*K+iNode[0][y]*(N/d1+(d1-1))+jNode[0][y]]=B[i*N+j];
        iNode[0][y]++;

    }
    for(int p=0;p<__NUMROW__;p++){
      for(int q=0;q<__NUMCOL__;q++){
        iNode[p][q]=0;
        if(q==y)
        jNode[p][q]++;
      }
    }
  }



  for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          C[i*ldC+j] = 0;
        }
      }

printf("Reached Here 06\n");
  redefine_initialize(__NUMROW__,
                      __NUMCOL__); // Create a REDEFINE fabric of size 1x1
  __re_StartHyperOpInit(0, 0, __NUMCOL__, __NUMROW__);
  __REDEFINE_main();  // Execute start hyperOp
  redefine_execute();     

  


  for(int i=0;i<__NUMROW__;i++){
    for(int j=0;j<__NUMCOL__;j++){
      x=0;
      for(int p=i;p<m;p+=d0){
        y=0;
        for(int q=j;q<n;q+=d1){
          C[p*n+q]=temp3[i*(M/d0+(d0-1))*(N/d1+(d1-1))*d1+j*(M/d0+(d0-1))*(N/d1+(d1-1))+x*(N/d1+(d1-1))+y];
          y++;
        }
        x++;
      }
    }
  }
  printf("Matrix C :\n");
  showMatrix(C, m, n);
  //printf("Matrix Cref :\n");
  //showMatrix(Cref, m, n);
	diff = MaxAbsDiff( m, n, C, ldC, Cref, ldC );
	printf( "%8.4le\n", diff  );


  printf("ALL OK WE ARE DONE\n");

  return 0;
}

int main(int argc, char* argv[])
{

/*  int n = 32;
  if(argc > 1){
    n = atoi(argv[1]);
    if(n%8 != 0){
      printf("INFO:Matrix size must be a multiple of 8, changing it to 32\n");
      n = 32;
    }
  }*/



  printf("#####################################################\n");
  printf("##########  TESTING MATRIX MULTIPLICATION  ##########\n");
  printf("#####################################################\n");

  runmatmul();

}
