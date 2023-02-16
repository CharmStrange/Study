#include <stdio.h>
#define Rows 3
#define Columns 3

int main() {
    int Matrix_A[Rows][Columns]={ {1,0,0}, {0,1,0}, {0,0,1} };
    /* 
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    */
    
    int Matrix_B[Rows][Columns]={ {0,1,1}, {1,0,1}, {1,1,0} };
    /* 
    [0, 1, 1]
    [1, 0, 1]
    [1, 1, 0]
    */
    
    int Matrix_C[Rows][Columns];
    int i,j;
    for(i=0; i<Rows; i++)
        for(j=0; j<Columns; j++)
            Matrix_C[i][j]=Matrix_A[i][j]+Matrix_B[i][j];

    // 체크
    /*for(i=0; i<Rows; i++)
       printf("%d", Matrix_A);*/
    printf("%d\n", sizeof(Matrix_A));
    
    /*for(i=0; i<Rows; i++)
       printf("%d", Matrix_A);*/
    printf("%d\n", sizeof(Matrix_B));
    
    /*for(i=0; i<Rows; i++)
       printf("%d", Matrix_A);*/
    printf("%d\n", sizeof(Matrix_C));
    return 0;
}