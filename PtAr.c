#include <stdio.h>
//배열 포인터, 포인터 배열
int main(void){
    int A[5]={1,2,3,4,5};
    int (*pA)[5];
    int i;
    pA=&A;
    return 0;

    /* 
    int *A[10]; 
    int (*A)[10];
    */
}