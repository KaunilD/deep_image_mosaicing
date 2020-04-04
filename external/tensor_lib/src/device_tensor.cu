#include "device_tensor.hpp"


struct cuda_deletor {
	void operator()(float* p) const {
		CHECK(cudaFree(p));
	}
};

/*********************************/
/************CONSTRUCTORS*********/
/*********************************/

template<int DIMS>
void device_tensor<DIMS>::allocate_data() {
	CHECK(cudaMalloc(&(this->m_data), this->get_n_elems() * sizeof(float)));
	this->m_data_ptr = std::shared_ptr<float>(this->m_data, cuda_deletor());
};

// vanilla constructor
template <int DIMS>
device_tensor<DIMS>::device_tensor(const std::array<int, DIMS> t_size) :tensor<DIMS>(t_size) {
	allocate_data();
};


// copy constructor: performs shallow copy
template <int DIMS>
device_tensor<DIMS>::device_tensor(const device_tensor<DIMS>& t_deviceTensor, bool copy=true) : tensor<DIMS>(t_deviceTensor.m_size) {
	allocate_data();
	if (copy) {
		this->copy(t_deviceTensor);
	}
};

template <int DIMS>
device_tensor<DIMS>::device_tensor(const host_tensor<DIMS>& t_hostTensor, bool copy=true) : tensor<DIMS>(t_hostTensor.m_size) {
	allocate_data();
	if (copy) {
		this->copy(t_hostTensor);
	}
};



/*********************************/
/**************HELPERS************/
/*********************************/

template <int DIMS>
void device_tensor<DIMS>::copy(const host_tensor<DIMS>& t_hostTensor) {
	assert(this->get_n_elems() == t_hostTensor.get_n_elems());
	CHECK(cudaMemcpy(this->get(), t_hostTensor.get(), this->get_n_elems() * sizeof(float), cudaMemcpyHostToDevice));
};

template <int DIMS>
void device_tensor<DIMS>::copy(const device_tensor<DIMS>& t_deviceTensor) {
	assert(this->get_n_elems() == t_deviceTensor.get_n_elems());
	CHECK(cudaMemcpy(this->get(), t_deviceTensor.get(), this->get_n_elems() * sizeof(float), cudaMemcpyDeviceToDevice));
};

/*********************************/
/********OPERATOR OVERLOADS*******/
/*********************************/

/*
template <int DIMS>
device_tensor<DIMS>& device_tensor<DIMS>::operator=(const device_tensor<DIMS>& t_deviceTensor) {
	copy(t_deviceTensor);
	return *this;
};
*/
/*********************************/
/*********INSTATNTIATIONS*********/
/*********************************/
template class device_tensor<1>;