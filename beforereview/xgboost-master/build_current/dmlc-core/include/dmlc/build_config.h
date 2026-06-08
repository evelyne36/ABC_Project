#ifndef DMLC_BUILD_CONFIG_H_
#define DMLC_BUILD_CONFIG_H_

/* #undef DMLC_FOPEN_64_PRESENT */

#if !defined(DMLC_FOPEN_64_PRESENT) && DMLC_USE_FOPEN64
  #define fopen64 std::fopen
#endif

#define DMLC_CXXABI_H_PRESENT
#define DMLC_EXECINFO_H_PRESENT

#if (defined DMLC_CXXABI_H_PRESENT) && (defined DMLC_EXECINFO_H_PRESENT)
  #ifndef DMLC_LOG_STACK_TRACE
  #define DMLC_LOG_STACK_TRACE 1
  #endif
  #ifndef DMLC_LOG_STACK_TRACE_SIZE
  #define DMLC_LOG_STACK_TRACE_SIZE 10
  #endif
  #define DMLC_EXECINFO_H <execinfo.h>
#endif

#define DMLC_NANOSLEEP_PRESENT

#define DMLC_CMAKE_LITTLE_ENDIAN 1

#define CMAKE_CURRENT_SOURCE_DIR "/Users/ancd/VSCodeProject/check_codereview0604cc/beforereview/xgboost-master/dmlc-core"

#endif  // DMLC_BUILD_CONFIG_H_

/* #undef DMLC_USE_PARQUET */
