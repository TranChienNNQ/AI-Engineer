/* Teaching CPU kernels; finite FP32 inputs, disjoint buffers. No threading/GPU. */
#include <stddef.h>
#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif
static NOINLINE void mul(const float *restrict x, float *restrict t, size_t n, float a) {
    for (size_t i=0;i<n;i++) t[i]=a*x[i];
}
static NOINLINE void add(const float *restrict t, float *restrict u, size_t n, float b) {
    for (size_t i=0;i<n;i++) u[i]=t[i]+b;
}
static NOINLINE void relu(const float *restrict u, float *restrict y, size_t n) {
    for (size_t i=0;i<n;i++) y[i]=u[i]>0.0f ? u[i] : 0.0f;
}
void separate(const float *restrict x, float *restrict y, float *restrict t,
              float *restrict u, size_t n, float a, float b) {
    mul(x,t,n,a); add(t,u,n,b); relu(u,y,n);
}
void fused(const float *restrict x, float *restrict y, size_t n, float a, float b) {
    for (size_t i=0;i<n;i++) {
        float z=a*x[i]+b;
        y[i]=z>0.0f ? z : 0.0f;
    }
}
