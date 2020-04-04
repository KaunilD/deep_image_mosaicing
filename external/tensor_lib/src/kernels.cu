#include "device_tensor.cuh"

__device__ const int BLOCK_WIDTH = 256;
__device__ const int BLOCK_HEIGHT = 8;
__device__ const int WARP_SIZE = 32;
__device__ const int MAX_THREADS = 1024;

/*
FILL KERNEL
This kernel will fill an allocatoin with a constant value 'val'.
*/
template<int N_DIMS>
__global__ void
kernel_fill_apply(device_tensor<N_DIMS> x, const float val) {
    size_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col >= x.get_n_elems()) return;

    x.at_linear(col) = val;
}

//GPU kernel wrapper for fill_apply.
template<int N_DIMS>
void fill_apply(device_tensor<N_DIMS> x, const float val) {
    dim3 blockSize(32 * 10);
    dim3 gridSize(1);

    gridSize.x = x.get_n_elems() / blockSize.x;

    kernel_fill_apply<N_DIMS> << <gridSize, blockSize >> > (x, val);
    cudaDeviceSynchronize();
}


/*
PPOINTWISE KERNEL
Will loop over elements of tensor x and y and apply an elementwise operation between them.
i.e. out[i][j] = op::op(x[i][j], y[i][j])
*/
template<typename op, int N_DIMS>
__global__ void
kernel_pointwise_apply(device_tensor<N_DIMS> out,
    const device_tensor<N_DIMS> x, const device_tensor<N_DIMS> y) {
    assert(x.get_n_elems() == y.get_n_elems());

    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    // threadId within the 2d thread block
    int threadId = threadIdx.x;

    extern __shared__ float sxsy[];

    if (col >= x.get_n_elems()) return;

    // first make 32 treads in the block copy 
    // data for the entire block of 32*32 threads
    if (threadId < 32) {
#pragma unroll
        // data for x
        for (int i = threadId; i < blockDim.x; i += 32) {
            sxsy[i] = x.at_linear(blockIdx.x * blockDim.x + i);
        }
        __syncthreads();
#pragma unroll
        // data for y.
        for (int i = threadId; i < blockDim.x; i += 32) {
            sxsy[i + blockDim.x] = y.at_linear(blockIdx.x * blockDim.x + i);
        }
        __syncthreads();
    }
    __syncthreads();

    // write to global
    out.at_linear(col) = op::op(sxsy[threadId], sxsy[blockDim.x + threadId]);
}

//GPU kernel wrapper for pointwise apply
template<typename op, int N_DIMS>
device_tensor<N_DIMS> pointwise_apply(const device_tensor<N_DIMS>& x, const device_tensor<N_DIMS>& y) {
    assert(x.get_n_elems() == y.get_n_elems());
    device_tensor<N_DIMS> out(x.size);

    dim3 blockSize(32 * 32, 1, 1);
    dim3 gridSize(1);
    // shared memory for x and y in the kernel
    size_t sMemBytes = 2 * sizeof(float) * blockSize.x;

    gridSize.x = x.get_n_elems() / blockSize.x;

    kernel_pointwise_apply<op, N_DIMS> << <gridSize, blockSize, sMemBytes >> > (out, x, y);
    cudaDeviceSynchronize();

    return out;
}

/*
PPOINTWISE KERNEL
Will loop over elements of tensor x and apply an elementwise operation on it.
i.e. out[i][j] = op::op(x[i][j])
*/
template<typename op, int N_DIMS>
__global__ void
kernel_pointwise_apply(device_tensor<N_DIMS> out,
    const device_tensor<N_DIMS> x) {

    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    int threadId = threadIdx.x;

    extern __shared__ float sx[]; // blockSize

    if (col >= x.get_n_elems()) return;

    // first make 32 treads in the block copy 
    // data for the entire block of 32*32 threads
    if (threadId < 32) {
#pragma unroll
        for (int i = threadId; i < blockDim.x; i += 32) {
            sx[i] = x.at_linear(blockIdx.x * blockDim.x + i);
        }
        __syncthreads();
    }
    __syncthreads();

    out.at_linear(col) = op::op(sx[threadId]);

}

//GPU kernel wrapper for pointwise apply
template<typename op, int N_DIMS>
device_tensor<N_DIMS> pointwise_apply(const device_tensor<N_DIMS>& x)
{
    device_tensor<N_DIMS> out(x.size);

    dim3 blockSize(320, 1, 1);
    dim3 gridSize(1);

    gridSize.x = x.get_n_elems() / blockSize.x;

    kernel_pointwise_apply<op, N_DIMS> << <gridSize, blockSize, sizeof(float)* blockSize.x >> > (out, x);
    cudaDeviceSynchronize();

    return out;
}


/* REDUCTION KERNEL
Takes a 2D tensor and reduces its second dimension based on op::op.
i.e. out[i] = init_value
     for j in J:
         out[i] = op::op(out[i], A[i, j])

Please see the Report.md for implementation details.
*/
template<typename op>
__global__ void  reduce_dim_partial(device_tensor<2> out,
    const device_tensor<2> in) {
    // 32*4bytes = 128bytes of 16K bytes 
    extern __shared__ float s[];
    // 2D array pixel coords
    int row = blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int threadId = threadIdx.x;
    // linear index in the 2D array
    if (col >= in.size[1] || row >= in.size[0]) return;

    s[threadId] = in.at(row, col);
    __syncthreads();

#pragma unroll
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (stride > threadId)
            s[threadId] = op::op(s[threadId], s[threadId + stride]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out.at(row, blockIdx.x) = s[0];
}
// quick hack for writing results into global 1 d tensor.
template<typename op>
__global__ void  reduce_dim_partial(device_tensor<1> out,
    const device_tensor<2> in) {
    // 32*4bytes = 128bytes of 16K bytes 
    extern __shared__ float s[];
    // 2D array pixel coords
    int row = blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int threadId = threadIdx.x;
    // linear index in the 2D array
    if (col >= in.size[1] || row >= in.size[0]) return;

    s[threadId] = in.at(row, col);
    __syncthreads();

#pragma unroll
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (stride > threadIdx.x)
            s[threadId] = op::op(s[threadId], s[threadId + stride]);
        __syncthreads();
    }
    if (threadId == 0) out.at(row) = s[0];
}

//GPU kernel wrapper for reduce dim=1
template<typename op>
device_tensor<1> reduce_apply(const device_tensor<2>& x)
{
    unsigned int gridx{ 0 }, gridy{ 0 };
    dim3 blockSize, gridSize;
    gridy = x.size[0];
    gridx = x.size[1] / (BLOCK_WIDTH);

    blockSize = dim3(BLOCK_WIDTH, 1, 1);
    gridSize = dim3(gridx, gridy, 1);

    // step1.  reduce K blocks to 1 column
    device_tensor<2> partial_out({ x.size[0], x.size[1] / blockSize.x });
    reduce_dim_partial<op> << <gridSize, blockSize, sizeof(float)* blockSize.x >> > (partial_out, x);

    cudaDeviceSynchronize();

    // step2. reduce K columns to 1 column
    blockSize.x = partial_out.size[1];
    gridSize.x = 1;

    device_tensor<1> out({ x.size[0] });
    reduce_dim_partial<op> << <gridSize, blockSize, sizeof(float)* blockSize.x >> > (out, partial_out);

    cudaDeviceSynchronize();

    return out;
}

/* BROADCAST KERNELS */
//Broadcasts tensor and applies to another in element wise fasion.
//i.e. out [i, j] = op::op(A[i] + B[i, j])
template<typename op>
__global__ void
kernel_broadcast_apply(device_tensor<2> out,
    const device_tensor<1> x, const device_tensor<2> y) {

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    // threadId within the 2d thread block
    int threadId = threadIdx.y + threadIdx.x * blockDim.y;

    extern __shared__ float s[];
    if (row >= y.size[0] || col >= y.size[1]) return;
    // load A[i] in shared mem
    if (threadId < blockDim.y) {
        s[threadIdx.y] = x.at(row);
    }
    __syncthreads();

    out.at(row, col) = op::op(s[threadIdx.y], y.at(row, col));

}

/* BROADCAST KERNELS */
//Broadcasts tensor and applies to another in element wise fasion.
//i.e. out [i, j] = op::op(A[i, j] + B[i])

// P.S. Report section Remarks: Point 4.
template<typename op>
__global__ void
kernel_broadcast_apply(device_tensor<2> out,
    const device_tensor<2> x, const device_tensor<1> y) {

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    // threadId within the 2d thread block
    int threadId = threadIdx.y + threadIdx.x * blockDim.y;

    extern __shared__ float s[];
    if (row >= x.size[0] || col >= x.size[1]) return;
    // load B[i] in shared mem
    if (threadId < blockDim.y) {
        s[threadIdx.y] = y.at(row);
    }
    __syncthreads();

    out.at(row, col) = op::op(x.at(row, col), s[threadIdx.y]);
    //out.at(row, col) = op::op(x.at(row, col), y.at(row));

}

//GPU kernel wrapper for first broadcast kernel
template<typename op>
device_tensor< 2 > broadcast_apply(const device_tensor<2>& x, const device_tensor<1>& y) {
    assert(x.size[0] == y.get_n_elems());
    device_tensor<2> out(x.size);

    dim3 blockSize(256, 8, 1);
    dim3 gridSize(1);

    gridSize.x = x.size[1] / blockSize.x;
    gridSize.y = x.size[0] / blockSize.y;

    kernel_broadcast_apply<op> << <gridSize, blockSize, blockSize.x * sizeof(float) >> > (out, x, y);
    cudaDeviceSynchronize();

    return out;
}

//GPU kernel wrapper for second broadcast kernel
template<typename op>
device_tensor< 2 > broadcast_apply(const device_tensor<1>& x, const device_tensor<2>& y) {
    assert(x.get_n_elems() == y.size[0]);
    device_tensor<2> out(y.size);

    dim3 blockSize(256, 8, 1);
    dim3 gridSize(1);

    gridSize.x = y.size[1] / blockSize.x;
    gridSize.y = y.size[0] / blockSize.y;

    kernel_broadcast_apply<op> << <gridSize, blockSize, blockSize.x * sizeof(float) >> > (out, x, y);
    cudaDeviceSynchronize();

    return out;
}