#pragma once
// Keep the upstream LAPACK DGESV algorithm; substitute the platform LAPACK ABI.
#include <cstdlib>
#include <cstring>
#include <cstddef>
extern "C" void dgesv_(int*,int*,double*,int*,int*,double*,int*,int*);
inline void dgesv(int* n,int* nrhs,double* a,int* lda,int* piv,double* b,int* ldb,int* info) {
    dgesv_(n,nrhs,a,lda,piv,b,ldb,info);
}
inline void* mkl_calloc(std::size_t n,std::size_t size,int alignment) {
    void* p=nullptr;
    if(posix_memalign(&p,alignment,n*size)!=0) return nullptr;
    std::memset(p,0,n*size); return p;
}
