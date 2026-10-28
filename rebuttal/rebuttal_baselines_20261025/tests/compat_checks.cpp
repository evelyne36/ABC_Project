#include <cassert>
#include <cmath>
#include <cstdint>
#include "x86intrin.h"
#include "mkl.h"

int main() {
    alignas(32) double a[4] = {1, 2, 3, 4};
    alignas(32) double b[4] = {4, 3, 2, 1};
    double c[4];
    _mm256_storeu_pd(c, _mm256_add_pd(_mm256_load_pd(a), _mm256_load_pd(b)));
    for (int i=0; i<4; ++i) assert(c[i] == 5);
    float f[8] = {1,2,3,4,5,6,7,8};
    auto packed = _mm256_loadu_ps(f);
    _mm256_storeu_pd(c, _mm256_cvtps_pd(_mm256_extractf128_ps(packed, 1)));
    assert(c[0] == 5 && c[3] == 8);
    double d[2];
    _mm_store_pd(d, _mm256_extractf128_pd(_mm256_load_pd(a), 1));
    assert(d[0] == 3 && d[1] == 4);
    assert(_pext_u64(53, 43) == 9);
    assert(_mm_popcnt_u64(0xf0f) == 8);
    auto p = static_cast<double*>(mkl_calloc(7, sizeof(double), 64));
    assert(p && reinterpret_cast<std::uintptr_t>(p) % 64 == 0);
    for (int i=0; i<7; ++i) assert(p[i] == 0);
    _mm_free(p);
    int n=2, nrhs=1, info=99, pivots[2];
    double matrix[4] = {3,1,1,2}, rhs[2] = {9,8};
    dgesv(&n,&nrhs,matrix,&n,pivots,rhs,&n,&info);
    assert(info == 0 && std::abs(rhs[0]-2)<1e-12 && std::abs(rhs[1]-3)<1e-12);
}
