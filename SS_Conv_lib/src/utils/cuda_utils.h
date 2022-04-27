// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

#endif
