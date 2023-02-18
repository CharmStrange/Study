#include<stdio.h>

struct vector{
    float x;
    float y;
}; // 구조체 vector, 실수형 x 와 y

struct vector get_vector_sum(struct vector a, struct vector b);

int main(void){
    struct vector a={1.0, 2.0};
    struct vector a={3.0, 4.0};
    struct vector sum;
    
    sum=get_vector_sum(a,b);
    //printf("%f, %f", sum.x, sum.y);
    return 0;
}

struct vector get_vector_sum(struct vector a, struct vector b){
    struct vector result;
    result.x=a.x+b.x;
    result.y=a.y+b.y;
    return result;
}