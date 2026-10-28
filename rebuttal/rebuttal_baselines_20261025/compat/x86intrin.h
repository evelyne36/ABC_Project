#pragma once
// Scalar equivalents of the exact Intel intrinsics used by upstream GBDT-PL.
// Local ARM correctness checks only; x86 builds retain the original intrinsics.
#if defined(__x86_64__) && !defined(GBDTPL_FORCE_SCALAR)
#include_next <x86intrin.h>
#else
#include <cstdint>
#include <cstdlib>
#include <cstring>
struct alignas(16) __m128 { float v[4]; };
struct alignas(16) __m128d { double v[2]; };
struct alignas(16) __m128i { std::int32_t v[4]; };
struct alignas(32) __m256 { float v[8]; };
struct alignas(32) __m256d { double v[4]; };
struct alignas(32) __m256i { std::int32_t v[8]; };
inline __m256d _mm256_load_pd(const double* p) { __m256d r; std::memcpy(&r,p,32); return r; }
inline __m256d _mm256_loadu_pd(const double* p) { return _mm256_load_pd(p); }
inline __m256 _mm256_load_ps(const float* p) { __m256 r; std::memcpy(&r,p,32); return r; }
inline __m256 _mm256_loadu_ps(const float* p) { return _mm256_load_ps(p); }
inline __m128 _mm_loadu_ps(const float* p) { __m128 r; std::memcpy(&r,p,16); return r; }
inline __m128d _mm_load_pd(const double* p) { __m128d r; std::memcpy(&r,p,16); return r; }
inline void _mm256_store_pd(double* p, __m256d x) { std::memcpy(p,&x,32); }
inline void _mm256_storeu_pd(double* p, __m256d x) { _mm256_store_pd(p,x); }
inline void _mm_store_pd(double* p, __m128d x) { std::memcpy(p,&x,16); }
inline __m256d _mm256_add_pd(__m256d a, __m256d b) { for(int i=0;i<4;++i) a.v[i]+=b.v[i]; return a; }
inline __m256d _mm256_sub_pd(__m256d a, __m256d b) { for(int i=0;i<4;++i) a.v[i]-=b.v[i]; return a; }
inline __m128d _mm_add_pd(__m128d a, __m128d b) { for(int i=0;i<2;++i) a.v[i]+=b.v[i]; return a; }
inline __m256d _mm256_cvtps_pd(__m128 a) { __m256d r; for(int i=0;i<4;++i) r.v[i]=a.v[i]; return r; }
inline __m128 _mm256_extractf128_ps(__m256 a, int half) { __m128 r; std::memcpy(&r,a.v+4*half,16); return r; }
inline __m128d _mm256_extractf128_pd(__m256d a, int half) { __m128d r; std::memcpy(&r,a.v+2*half,16); return r; }
inline __m128i _mm_set_epi32(int a3,int a2,int a1,int a0) { return {{a0,a1,a2,a3}}; }
inline __m256i _mm256_set_epi32(int a7,int a6,int a5,int a4,int a3,int a2,int a1,int a0) { return {{a0,a1,a2,a3,a4,a5,a6,a7}}; }
inline __m256i _mm256_set_epi64x(std::int64_t a3,std::int64_t a2,std::int64_t a1,std::int64_t a0) {
    std::int64_t v[4]={a0,a1,a2,a3}; __m256i r; std::memcpy(&r,v,32); return r;
}
inline int _mm_popcnt_u64(std::uint64_t a) { return __builtin_popcountll(a); }
inline std::uint64_t _pext_u64(std::uint64_t bits,std::uint64_t mask) {
    std::uint64_t out=0, next=1;
    while(mask) { auto low=mask & (0-mask); if(bits & low) out|=next; mask&=mask-1; next<<=1; }
    return out;
}
inline void* _mm_malloc(std::size_t n,std::size_t alignment) {
    void* p=nullptr; return posix_memalign(&p,alignment,n)==0 ? p : nullptr;
}
inline void _mm_free(void* p) { std::free(p); }
#endif
