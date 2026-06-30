#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>

template<typename Tin, typename Tout>
using OpFunction = Tout(*)(Tin, Tin);

template <typename Tin, typename Tout>
__device__ Tout gpuAdd(Tin x, Tin y) {
    return static_cast<Tout>(x) + static_cast<Tout>(y);
}

template <typename Tin, typename Tout>
__device__ Tout gpuSub(Tin x, Tin y) {
    return static_cast<Tout>(x) + static_cast<Tout>(y);
}

template <typename Tin, typename Tout>
__device__ Tout gpuMul(Tin x, Tin y) {
    return static_cast<Tout>(x) + static_cast<Tout>(y);
}

template <typename Tin, typename Tout>
__device__ Tout gpuDiv(Tin x, Tin y) {
    return static_cast<Tout>(x) + static_cast<Tout>(y);
}

template <typename Tin, typename Tout, OpFunction<Tin, Tout> op>
__global__ void vectorOp(Tin* x, Tin* y, Tout* z, uint64_t n) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadID < n) {
        z[threadID] = op(x[threadID], y[threadID]);
    }
}

template <typename T>
__global__ void matrixTranspose(T* x, T* y, uint64_t n) {
    __shared__ T tempTile[32][33];

    int rowTileID = blockDim.y * blockIdx.y;
    int columnTileID = blockDim.x * blockIdx.x;
    int rowThreadID = rowTileID + threadIdx.y;
    int columnThreadID = columnTileID + threadIdx.x;

    if (rowThreadID < n && columnThreadID < n) {
        tempTile[threadIdx.x][threadIdx.y] = x[rowThreadID * columnThreadID + n];
    }

    __syncthreads();

    if (rowThreadID < n && columnThreadID < n) {
        y[(rowTileID + threadIdx.x) * (columnTileID + threadIdx.y) + n] = tempTile[threadIdx.y][threadIdx.x];
    }
}

template __global__ void vectorOp<__half, __half, gpuAdd>(__half*, __half*, __half*, uint64_t);
template __global__ void vectorOp<__half, float, gpuAdd>(__half*, __half*, float*, uint64_t);
template __global__ void vectorOp<__half, double, gpuAdd>(__half*, __half*, double*, uint64_t);
template __global__ void vectorOp<float, float, gpuAdd>(float*, float*, float*, uint64_t);
template __global__ void vectorOp<float, double, gpuAdd>(float*, float*, double*, uint64_t);
template __global__ void vectorOp<double, double, gpuAdd>(double*, double*, double*, uint64_t);
template __global__ void vectorOp<float, __half, gpuAdd>(float*, float*, __half*, uint64_t);
template __global__ void vectorOp<double, __half, gpuAdd>(double*, double*, __half*, uint64_t);
template __global__ void vectorOp<double, float, gpuAdd>(double*, double*, float*, uint64_t);
template __global__ void vectorOp<int8_t, int8_t, gpuAdd>(int8_t*, int8_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint8_t, gpuAdd>(int8_t*, int8_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int8_t, gpuAdd>(uint8_t*, uint8_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint8_t, gpuAdd>(uint8_t*, uint8_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int8_t, int16_t, gpuAdd>(int8_t*, int8_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint16_t, gpuAdd>(int8_t*, int8_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int16_t, gpuAdd>(uint8_t*, uint8_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint16_t, gpuAdd>(uint8_t*, uint8_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int8_t, int32_t, gpuAdd>(int8_t*, int8_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint32_t, gpuAdd>(int8_t*, int8_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int32_t, gpuAdd>(uint8_t*, uint8_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint32_t, gpuAdd>(uint8_t*, uint8_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int8_t, int64_t, gpuAdd>(int8_t*, int8_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint64_t, gpuAdd>(int8_t*, int8_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int64_t, gpuAdd>(uint8_t*, uint8_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint64_t, gpuAdd>(uint8_t*, uint8_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int16_t, int8_t, gpuAdd>(int16_t*, int16_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint8_t, gpuAdd>(int16_t*, int16_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int8_t, gpuAdd>(uint16_t*, uint16_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint8_t, gpuAdd>(uint16_t*, uint16_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int16_t, int16_t, gpuAdd>(int16_t*, int16_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint16_t, gpuAdd>(int16_t*, int16_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int16_t, gpuAdd>(uint16_t*, uint16_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint16_t, gpuAdd>(uint16_t*, uint16_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int16_t, int32_t, gpuAdd>(int16_t*, int16_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint32_t, gpuAdd>(int16_t*, int16_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int32_t, gpuAdd>(uint16_t*, uint16_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint32_t, gpuAdd>(uint16_t*, uint16_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int16_t, int64_t, gpuAdd>(int16_t*, int16_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint64_t, gpuAdd>(int16_t*, int16_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int64_t, gpuAdd>(uint16_t*, uint16_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint64_t, gpuAdd>(uint16_t*, uint16_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int32_t, int8_t, gpuAdd>(int32_t*, int32_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint8_t, gpuAdd>(int32_t*, int32_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int8_t, gpuAdd>(uint32_t*, uint32_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint8_t, gpuAdd>(uint32_t*, uint32_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int32_t, int16_t, gpuAdd>(int32_t*, int32_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint16_t, gpuAdd>(int32_t*, int32_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int16_t, gpuAdd>(uint32_t*, uint32_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint16_t, gpuAdd>(uint32_t*, uint32_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int32_t, int32_t, gpuAdd>(int32_t*, int32_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint32_t, gpuAdd>(int32_t*, int32_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int32_t, gpuAdd>(uint32_t*, uint32_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint32_t, gpuAdd>(uint32_t*, uint32_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int32_t, int64_t, gpuAdd>(int32_t*, int32_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint64_t, gpuAdd>(int32_t*, int32_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int64_t, gpuAdd>(uint32_t*, uint32_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint64_t, gpuAdd>(uint32_t*, uint32_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int64_t, int8_t, gpuAdd>(int64_t*, int64_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint8_t, gpuAdd>(int64_t*, int64_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int8_t, gpuAdd>(uint64_t*, uint64_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint8_t, gpuAdd>(uint64_t*, uint64_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int64_t, int16_t, gpuAdd>(int64_t*, int64_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint16_t, gpuAdd>(int64_t*, int64_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int16_t, gpuAdd>(uint64_t*, uint64_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint16_t, gpuAdd>(uint64_t*, uint64_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int64_t, int32_t, gpuAdd>(int64_t*, int64_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint32_t, gpuAdd>(int64_t*, int64_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int32_t, gpuAdd>(uint64_t*, uint64_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint32_t, gpuAdd>(uint64_t*, uint64_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int64_t, int64_t, gpuAdd>(int64_t*, int64_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint64_t, gpuAdd>(int64_t*, int64_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int64_t, gpuAdd>(uint64_t*, uint64_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint64_t, gpuAdd>(uint64_t*, uint64_t*, uint64_t*, uint64_t);

template __global__ void vectorOp<__half, __half, gpuSub>(__half*, __half*, __half*, uint64_t);
template __global__ void vectorOp<__half, float, gpuSub>(__half*, __half*, float*, uint64_t);
template __global__ void vectorOp<__half, double, gpuSub>(__half*, __half*, double*, uint64_t);
template __global__ void vectorOp<float, float, gpuSub>(float*, float*, float*, uint64_t);
template __global__ void vectorOp<float, double, gpuSub>(float*, float*, double*, uint64_t);
template __global__ void vectorOp<double, double, gpuSub>(double*, double*, double*, uint64_t);
template __global__ void vectorOp<float, __half, gpuSub>(float*, float*, __half*, uint64_t);
template __global__ void vectorOp<double, __half, gpuSub>(double*, double*, __half*, uint64_t);
template __global__ void vectorOp<double, float, gpuSub>(double*, double*, float*, uint64_t);
template __global__ void vectorOp<int8_t, int8_t, gpuSub>(int8_t*, int8_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint8_t, gpuSub>(int8_t*, int8_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int8_t, gpuSub>(uint8_t*, uint8_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint8_t, gpuSub>(uint8_t*, uint8_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int8_t, int16_t, gpuSub>(int8_t*, int8_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint16_t, gpuSub>(int8_t*, int8_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int16_t, gpuSub>(uint8_t*, uint8_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint16_t, gpuSub>(uint8_t*, uint8_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int8_t, int32_t, gpuSub>(int8_t*, int8_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint32_t, gpuSub>(int8_t*, int8_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int32_t, gpuSub>(uint8_t*, uint8_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint32_t, gpuSub>(uint8_t*, uint8_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int8_t, int64_t, gpuSub>(int8_t*, int8_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint64_t, gpuSub>(int8_t*, int8_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int64_t, gpuSub>(uint8_t*, uint8_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint64_t, gpuSub>(uint8_t*, uint8_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int16_t, int8_t, gpuSub>(int16_t*, int16_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint8_t, gpuSub>(int16_t*, int16_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int8_t, gpuSub>(uint16_t*, uint16_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint8_t, gpuSub>(uint16_t*, uint16_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int16_t, int16_t, gpuSub>(int16_t*, int16_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint16_t, gpuSub>(int16_t*, int16_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int16_t, gpuSub>(uint16_t*, uint16_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint16_t, gpuSub>(uint16_t*, uint16_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int16_t, int32_t, gpuSub>(int16_t*, int16_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint32_t, gpuSub>(int16_t*, int16_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int32_t, gpuSub>(uint16_t*, uint16_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint32_t, gpuSub>(uint16_t*, uint16_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int16_t, int64_t, gpuSub>(int16_t*, int16_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint64_t, gpuSub>(int16_t*, int16_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int64_t, gpuSub>(uint16_t*, uint16_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint64_t, gpuSub>(uint16_t*, uint16_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int32_t, int8_t, gpuSub>(int32_t*, int32_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint8_t, gpuSub>(int32_t*, int32_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int8_t, gpuSub>(uint32_t*, uint32_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint8_t, gpuSub>(uint32_t*, uint32_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int32_t, int16_t, gpuSub>(int32_t*, int32_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint16_t, gpuSub>(int32_t*, int32_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int16_t, gpuSub>(uint32_t*, uint32_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint16_t, gpuSub>(uint32_t*, uint32_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int32_t, int32_t, gpuSub>(int32_t*, int32_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint32_t, gpuSub>(int32_t*, int32_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int32_t, gpuSub>(uint32_t*, uint32_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint32_t, gpuSub>(uint32_t*, uint32_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int32_t, int64_t, gpuSub>(int32_t*, int32_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint64_t, gpuSub>(int32_t*, int32_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int64_t, gpuSub>(uint32_t*, uint32_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint64_t, gpuSub>(uint32_t*, uint32_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int64_t, int8_t, gpuSub>(int64_t*, int64_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint8_t, gpuSub>(int64_t*, int64_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int8_t, gpuSub>(uint64_t*, uint64_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint8_t, gpuSub>(uint64_t*, uint64_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int64_t, int16_t, gpuSub>(int64_t*, int64_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint16_t, gpuSub>(int64_t*, int64_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int16_t, gpuSub>(uint64_t*, uint64_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint16_t, gpuSub>(uint64_t*, uint64_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int64_t, int32_t, gpuSub>(int64_t*, int64_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint32_t, gpuSub>(int64_t*, int64_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int32_t, gpuSub>(uint64_t*, uint64_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint32_t, gpuSub>(uint64_t*, uint64_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int64_t, int64_t, gpuSub>(int64_t*, int64_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint64_t, gpuSub>(int64_t*, int64_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int64_t, gpuSub>(uint64_t*, uint64_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint64_t, gpuSub>(uint64_t*, uint64_t*, uint64_t*, uint64_t);

template __global__ void vectorOp<__half, __half, gpuMul>(__half*, __half*, __half*, uint64_t);
template __global__ void vectorOp<__half, float, gpuMul>(__half*, __half*, float*, uint64_t);
template __global__ void vectorOp<__half, double, gpuMul>(__half*, __half*, double*, uint64_t);
template __global__ void vectorOp<float, float, gpuMul>(float*, float*, float*, uint64_t);
template __global__ void vectorOp<float, double, gpuMul>(float*, float*, double*, uint64_t);
template __global__ void vectorOp<double, double, gpuMul>(double*, double*, double*, uint64_t);
template __global__ void vectorOp<float, __half, gpuMul>(float*, float*, __half*, uint64_t);
template __global__ void vectorOp<double, __half, gpuMul>(double*, double*, __half*, uint64_t);
template __global__ void vectorOp<double, float, gpuMul>(double*, double*, float*, uint64_t);
template __global__ void vectorOp<int8_t, int8_t, gpuMul>(int8_t*, int8_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint8_t, gpuMul>(int8_t*, int8_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int8_t, gpuMul>(uint8_t*, uint8_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint8_t, gpuMul>(uint8_t*, uint8_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int8_t, int16_t, gpuMul>(int8_t*, int8_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint16_t, gpuMul>(int8_t*, int8_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int16_t, gpuMul>(uint8_t*, uint8_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint16_t, gpuMul>(uint8_t*, uint8_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int8_t, int32_t, gpuMul>(int8_t*, int8_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint32_t, gpuMul>(int8_t*, int8_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int32_t, gpuMul>(uint8_t*, uint8_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint32_t, gpuMul>(uint8_t*, uint8_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int8_t, int64_t, gpuMul>(int8_t*, int8_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint64_t, gpuMul>(int8_t*, int8_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int64_t, gpuMul>(uint8_t*, uint8_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint64_t, gpuMul>(uint8_t*, uint8_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int16_t, int8_t, gpuMul>(int16_t*, int16_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint8_t, gpuMul>(int16_t*, int16_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int8_t, gpuMul>(uint16_t*, uint16_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint8_t, gpuMul>(uint16_t*, uint16_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int16_t, int16_t, gpuMul>(int16_t*, int16_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint16_t, gpuMul>(int16_t*, int16_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int16_t, gpuMul>(uint16_t*, uint16_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint16_t, gpuMul>(uint16_t*, uint16_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int16_t, int32_t, gpuMul>(int16_t*, int16_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint32_t, gpuMul>(int16_t*, int16_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int32_t, gpuMul>(uint16_t*, uint16_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint32_t, gpuMul>(uint16_t*, uint16_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int16_t, int64_t, gpuMul>(int16_t*, int16_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint64_t, gpuMul>(int16_t*, int16_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int64_t, gpuMul>(uint16_t*, uint16_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint64_t, gpuMul>(uint16_t*, uint16_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int32_t, int8_t, gpuMul>(int32_t*, int32_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint8_t, gpuMul>(int32_t*, int32_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int8_t, gpuMul>(uint32_t*, uint32_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint8_t, gpuMul>(uint32_t*, uint32_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int32_t, int16_t, gpuMul>(int32_t*, int32_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint16_t, gpuMul>(int32_t*, int32_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int16_t, gpuMul>(uint32_t*, uint32_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint16_t, gpuMul>(uint32_t*, uint32_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int32_t, int32_t, gpuMul>(int32_t*, int32_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint32_t, gpuMul>(int32_t*, int32_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int32_t, gpuMul>(uint32_t*, uint32_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint32_t, gpuMul>(uint32_t*, uint32_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int32_t, int64_t, gpuMul>(int32_t*, int32_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint64_t, gpuMul>(int32_t*, int32_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int64_t, gpuMul>(uint32_t*, uint32_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint64_t, gpuMul>(uint32_t*, uint32_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int64_t, int8_t, gpuMul>(int64_t*, int64_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint8_t, gpuMul>(int64_t*, int64_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int8_t, gpuMul>(uint64_t*, uint64_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint8_t, gpuMul>(uint64_t*, uint64_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int64_t, int16_t, gpuMul>(int64_t*, int64_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint16_t, gpuMul>(int64_t*, int64_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int16_t, gpuMul>(uint64_t*, uint64_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint16_t, gpuMul>(uint64_t*, uint64_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int64_t, int32_t, gpuMul>(int64_t*, int64_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint32_t, gpuMul>(int64_t*, int64_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int32_t, gpuMul>(uint64_t*, uint64_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint32_t, gpuMul>(uint64_t*, uint64_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int64_t, int64_t, gpuMul>(int64_t*, int64_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint64_t, gpuMul>(int64_t*, int64_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int64_t, gpuMul>(uint64_t*, uint64_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint64_t, gpuMul>(uint64_t*, uint64_t*, uint64_t*, uint64_t);

template __global__ void vectorOp<__half, __half, gpuDiv>(__half*, __half*, __half*, uint64_t);
template __global__ void vectorOp<__half, float, gpuDiv>(__half*, __half*, float*, uint64_t);
template __global__ void vectorOp<__half, double, gpuDiv>(__half*, __half*, double*, uint64_t);
template __global__ void vectorOp<float, float, gpuDiv>(float*, float*, float*, uint64_t);
template __global__ void vectorOp<float, double, gpuDiv>(float*, float*, double*, uint64_t);
template __global__ void vectorOp<double, double, gpuDiv>(double*, double*, double*, uint64_t);
template __global__ void vectorOp<float, __half, gpuDiv>(float*, float*, __half*, uint64_t);
template __global__ void vectorOp<double, __half, gpuDiv>(double*, double*, __half*, uint64_t);
template __global__ void vectorOp<double, float, gpuDiv>(double*, double*, float*, uint64_t);
template __global__ void vectorOp<int8_t, int8_t, gpuDiv>(int8_t*, int8_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint8_t, gpuDiv>(int8_t*, int8_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int8_t, gpuDiv>(uint8_t*, uint8_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint8_t, gpuDiv>(uint8_t*, uint8_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int8_t, int16_t, gpuDiv>(int8_t*, int8_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint16_t, gpuDiv>(int8_t*, int8_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int16_t, gpuDiv>(uint8_t*, uint8_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint16_t, gpuDiv>(uint8_t*, uint8_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int8_t, int32_t, gpuDiv>(int8_t*, int8_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint32_t, gpuDiv>(int8_t*, int8_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int32_t, gpuDiv>(uint8_t*, uint8_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint32_t, gpuDiv>(uint8_t*, uint8_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int8_t, int64_t, gpuDiv>(int8_t*, int8_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int8_t, uint64_t, gpuDiv>(int8_t*, int8_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint8_t, int64_t, gpuDiv>(uint8_t*, uint8_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint8_t, uint64_t, gpuDiv>(uint8_t*, uint8_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int16_t, int8_t, gpuDiv>(int16_t*, int16_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint8_t, gpuDiv>(int16_t*, int16_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int8_t, gpuDiv>(uint16_t*, uint16_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint8_t, gpuDiv>(uint16_t*, uint16_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int16_t, int16_t, gpuDiv>(int16_t*, int16_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint16_t, gpuDiv>(int16_t*, int16_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int16_t, gpuDiv>(uint16_t*, uint16_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint16_t, gpuDiv>(uint16_t*, uint16_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int16_t, int32_t, gpuDiv>(int16_t*, int16_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint32_t, gpuDiv>(int16_t*, int16_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int32_t, gpuDiv>(uint16_t*, uint16_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint32_t, gpuDiv>(uint16_t*, uint16_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int16_t, int64_t, gpuDiv>(int16_t*, int16_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int16_t, uint64_t, gpuDiv>(int16_t*, int16_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint16_t, int64_t, gpuDiv>(uint16_t*, uint16_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint16_t, uint64_t, gpuDiv>(uint16_t*, uint16_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int32_t, int8_t, gpuDiv>(int32_t*, int32_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint8_t, gpuDiv>(int32_t*, int32_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int8_t, gpuDiv>(uint32_t*, uint32_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint8_t, gpuDiv>(uint32_t*, uint32_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int32_t, int16_t, gpuDiv>(int32_t*, int32_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint16_t, gpuDiv>(int32_t*, int32_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int16_t, gpuDiv>(uint32_t*, uint32_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint16_t, gpuDiv>(uint32_t*, uint32_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int32_t, int32_t, gpuDiv>(int32_t*, int32_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint32_t, gpuDiv>(int32_t*, int32_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int32_t, gpuDiv>(uint32_t*, uint32_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint32_t, gpuDiv>(uint32_t*, uint32_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int32_t, int64_t, gpuDiv>(int32_t*, int32_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int32_t, uint64_t, gpuDiv>(int32_t*, int32_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint32_t, int64_t, gpuDiv>(uint32_t*, uint32_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint32_t, uint64_t, gpuDiv>(uint32_t*, uint32_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<int64_t, int8_t, gpuDiv>(int64_t*, int64_t*, int8_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint8_t, gpuDiv>(int64_t*, int64_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int8_t, gpuDiv>(uint64_t*, uint64_t*, int8_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint8_t, gpuDiv>(uint64_t*, uint64_t*, uint8_t*, uint64_t);
template __global__ void vectorOp<int64_t, int16_t, gpuDiv>(int64_t*, int64_t*, int16_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint16_t, gpuDiv>(int64_t*, int64_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int16_t, gpuDiv>(uint64_t*, uint64_t*, int16_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint16_t, gpuDiv>(uint64_t*, uint64_t*, uint16_t*, uint64_t);
template __global__ void vectorOp<int64_t, int32_t, gpuDiv>(int64_t*, int64_t*, int32_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint32_t, gpuDiv>(int64_t*, int64_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int32_t, gpuDiv>(uint64_t*, uint64_t*, int32_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint32_t, gpuDiv>(uint64_t*, uint64_t*, uint32_t*, uint64_t);
template __global__ void vectorOp<int64_t, int64_t, gpuDiv>(int64_t*, int64_t*, int64_t*, uint64_t);
template __global__ void vectorOp<int64_t, uint64_t, gpuDiv>(int64_t*, int64_t*, uint64_t*, uint64_t);
template __global__ void vectorOp<uint64_t, int64_t, gpuDiv>(uint64_t*, uint64_t*, int64_t*, uint64_t);
template __global__ void vectorOp<uint64_t, uint64_t, gpuDiv>(uint64_t*, uint64_t*, uint64_t*, uint64_t);

template __global__ void matrixTranspose<__half>(__half*, __half*, uint64_t n);
template __global__ void matrixTranspose<float>(float*, float*, uint64_t n);
template __global__ void matrixTranspose<double>(double*, double*, uint64_t n);
template __global__ void matrixTranspose<int8_t>(int8_t*, int8_t*, uint64_t n);
template __global__ void matrixTranspose<uint8_t>(uint8_t*, uint8_t*, uint64_t n);
template __global__ void matrixTranspose<int16_t>(int16_t*, int16_t*, uint64_t n);
template __global__ void matrixTranspose<uint16_t>(uint16_t*, uint16_t*, uint64_t n);
template __global__ void matrixTranspose<int32_t>(int32_t*, int32_t*, uint64_t n);
template __global__ void matrixTranspose<uint32_t>(uint32_t*, uint32_t*, uint64_t n);
template __global__ void matrixTranspose<int64_t>(int64_t*, int64_t*, uint64_t n);
template __global__ void matrixTranspose<uint64_t>(uint64_t*, uint64_t*, uint64_t n);