#include "tensor.hpp"

template<int DIMS>
__host__ __device__ 
float* tensor<DIMS>::get() const { 
	return m_data; 
};


template<int DIMS>
void tensor<DIMS>::set_n_elems() {
	m_num_elements = 1;
	for (int i = 0; i < m_size.size(); i++) {
		m_num_elements *= m_size[i];
	}
};

template<int DIMS>
tensor<DIMS>::tensor(const std::array<int, DIMS> t_size) : m_size(t_size) {
	static_assert(DIMS == 1, "Only 1D Tensors supported!");
	set_n_elems();
};

template<int DIMS>
__host__ __device__ 
size_t tensor<DIMS>::get_n_elems() const {
	return m_num_elements; 
};

template<int DIMS>
__host__ __device__
float& tensor<DIMS>::at(size_t x) const {
	return *(this->m_data + x);
};

template class tensor<1>;
