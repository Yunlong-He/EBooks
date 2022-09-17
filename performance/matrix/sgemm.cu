#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <time.h>

#define THREAD_NUM 256
#define MATRIX_SIZE 1000
const int blocks_num = MATRIX_SIZE * (MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;
void matgen(float *a, int n);

///生成随机矩阵
void matgen(float *a, int n){
    int i, j;

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            a[i * n + j] = (float)rand();
            //printf("%f ", a[i*n + j]);
        }
    }
}

//__global__函数  并行计算矩阵乘法
__global__ static void matMult(const float *a, const float *b, float * c,int n){
    //表示目前的thread是第几个thread(由0开始计算)
    const int tid = threadIdx.x;

    //表示目前的thread属于第几个block(由0开始计算)
    const int bid = blockIdx.x;

    //从bid和tid计算出这个thread应该计算的row和column
    const int idx = bid * THREAD_NUM + tid;
    const int row = idx / n;
    const int column = idx % n;

    //计算矩阵乘法
    if (row < n && column < n){
        float t = 0;
        for (int i = 0; i < n; i++){
            t += a[row * n + i] * b[i * n + column];
        }
        c[row * n + column] = t;
    }
}

int main(void){
    cudaEvent_t stop, start;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //定义矩阵
    float *a, *b, *c;
    int n = MATRIX_SIZE;
    float elapsedTime = 0;

    //分配内存
    a = (float *)malloc(sizeof(float) * n * n);
    b = (float *)malloc(sizeof(float) * n * n);
    c = (float *)malloc(sizeof(float) * n * n);

    //设置随机数种子
    srand(0);

    //随机生成矩阵
    matgen(a, n);
    matgen(b, n);

    //分配GPU内存
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * n * n);
    cudaMalloc((void**)&d_b, sizeof(float) * n * n);
    cudaMalloc((void**)&d_c, sizeof(float) * n * n);
    cudaMemcpy(d_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    matMult << <blocks_num, THREAD_NUM, 0 >> >(d_a, d_b, d_c,n);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%f\n", elapsedTime);

    cudaMemcpy(c, d_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    /*for (int i = 0; i < 100; i++){
        printf("%f ", c[i]);
    }*/

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}