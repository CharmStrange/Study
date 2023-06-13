#include<stdio.h>
#include <stdlib.h>
#include <time.h>

int get_integer(void);
int is_prime(int n);

int main(void){
    int n, result;
    n=get_integer();
    if(is_prime(n)==1)
        printf("YES%d", n);
    else
        printf("NO%d", n);
    return 0;
}

int get_integer(void){
    int n;
    srand((unsigned)time(NULL));
    n=rand()%20;
    return n;
}

int is_prime(int n){
    int i;
    for(i=2; i<n; i++)
        if(n%i==0)
            return 0;
    return 1;
}
