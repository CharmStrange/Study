#include<stdio.h>
#include <stdlib.h>
#include <time.h> // 추가된 부분
#define number 10

int main() {
    int array[number]={0};
    int i;
    srand((unsigned)time(NULL)); // 추가된 부분
    for(i=0; i<number; i++)
        array[i]=rand()%10;
    int num1=array[0];
    for(i=0; i<number; i++)
        if(array[i]<num1)
            array[i]=num1;
    printf("%d", num1);
    
    return 0;
}
