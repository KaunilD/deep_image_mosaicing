### tensor_lib
#### A smart pointer based wrapper for CUDA memory allocations in c++ 17.

#### Dependencies:

1. [CMake v3.8+](https://cmake.org/download/) [for CUDA support within CMake]
2. [CUDA v9.0+](https://developer.nvidia.com/cuda-92-download-archive) 

#### Build

```cmake
mkdir build
cd build
cmake ..
make install
```

#### Example usage:

1. To integrate the library in your code check out the [examples/CMakeLists.txt](https://github.com/KaunilD/tensor_lib/blob/master/examples/CMakeLists.txt)

```c++
#include "host_tensor.hpp"	// managed ptr to data stored on the CPU
#include "device_tensor.hpp"	// managed ptr to data stored on the GPU

....
    
int main(){
    std::array<int, 1> size{1024};
    
    /* create a 1D tensor holding 1024 floats.*/
    host_tensor</*DIMS=*/ 1> a(/*size=*/ size, /*rand=*/ true);
    
    /* copy data to device. */
    device_tensor<1> b(a, /*copy=*/ true);
    
    /* copy data from the device to host. */
    host_tensor<1> c(b, /*copy=*/ true);
    
   .... 
         
   /* access data at i-th location. */
   std::cout << device_tensor.at(i);	/* done in device code */
   std::cout << host_tensor.at(i);	/* done in host code */
}
```



Road Map

- [x] 1D support
- [x] 2D support
- [ ] 3D support
