#ifndef __HOST_TENSOR_H__
#define __HOST_TENSOR_H__

#include "tensor.hpp"
#include "device_tensor.hpp"

template<int DIMS>
class host_tensor : public tensor<DIMS> {
	friend device_tensor<DIMS>;

	/*	op is the operator from ops.hpp 
		to be applied to the elements of the
		tensor
	*/
	template<typename op>
	host_tensor<DIMS> binary_apply(const host_tensor<DIMS>&,  const host_tensor<DIMS>&);

protected:

	virtual void allocate_data();
	virtual void copy(const host_tensor<DIMS>&);
	virtual void copy(const device_tensor<DIMS>&);
public:
	host_tensor() :tensor<DIMS>() {};
	/* create a host_tensor with random values */
	host_tensor(const std::array<int, DIMS>, bool /*rand = true*/);
	/* create a host_tensor with specified value */
	host_tensor(const std::array<int, DIMS>, float /*val=0.0f*/ );

	/* copy constructor from the device_tensor, 
		second param is copy: if false then creates 
		a tensor with the same dimensions as the const reference
	*/
	host_tensor(const device_tensor<1>&, bool /* copy = true */);
	host_tensor(const host_tensor<1>&, bool /* copy = true */);
	
	/* implements copy and swap idiom. */
	host_tensor<DIMS>& operator=(const host_tensor<DIMS>&) = delete;
	
	/* arithmetic: does pointwise math */
	host_tensor<DIMS> operator+(const host_tensor<DIMS>&);
	host_tensor<DIMS> operator-(const host_tensor<DIMS>&);
	host_tensor<DIMS> operator/(const host_tensor<DIMS>&);
	host_tensor<DIMS> operator*(const host_tensor<DIMS>&);

	~host_tensor() = default;
	/* helpers */
	void fill_random();
	void fill(float );

};
#endif __HOST_TENSOR_H__