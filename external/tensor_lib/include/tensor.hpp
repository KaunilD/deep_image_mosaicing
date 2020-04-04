// tensor_lib.h : Include file for standard system include files,
// or project specific include files.
#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <array>
#include <memory>
#include <cassert>
#include <iostream>

#include "cuda_runtime.h"
#include "utils.hpp"
#include "ops.hpp"

template<int>
class device_tensor;

template<int>
class host_tensor;

template<int DIMS>
class tensor {


protected:
	size_t m_num_elements {0};			/* total number of floats held by this tensor */
	
	float* m_data{ nullptr };
	std::shared_ptr<float> m_data_ptr;	/* RAII on the m_data array */
	
	virtual void allocate_data() = 0;

	void set_n_elems();
	__host__ __device__  float* get() const;
public:
	const std::array<int, DIMS> m_size{0};		// number of elements in each dimension
	
	/* c_tors */
	tensor() = default;
	tensor(const std::array<int, DIMS> t_size);
	~tensor() = default;

	__host__ __device__ size_t get_n_elems() const;
	
	/*	Accessors to return elements 
		stored in ROW MAJOR order
	*/
	/* 1D access: returns this[x] */
	__host__ __device__ float& at(size_t x) const;
	
	virtual void copy(const host_tensor<DIMS>&)		= 0;
	virtual void copy(const device_tensor<DIMS>&)	= 0;

};
#endif