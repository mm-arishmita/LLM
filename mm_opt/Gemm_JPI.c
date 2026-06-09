


void MyGemm( int m, int n, int k, double *A, int ldA,
	     double *B, int ldB, double *C, int ldC )
{
  for ( int i=0; i<m; i++ )
    for ( int j=0; j<n; j++ )
      for ( int p=0; p<k; p++ )
        C[ (i)*ldC + j ] += A[ (i)*ldA + p ] * B[ (p)*ldB + j ];
}
  