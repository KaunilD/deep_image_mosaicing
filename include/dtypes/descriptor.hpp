#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include "libs.hpp"

struct Descriptor {
public:
	MatrixXf_RM m_descp;
	
	int id;

	Descriptor() = default;


	Descriptor(const torch::Tensor& t_descp) {
		assert(t_descp.sizes().size() == 1 && t_descp.size(0) > 128);

		m_descp = MatrixXf_RM(1, static_cast<int>(t_descp.size(0)));
		std::copy(
			t_descp.data<float>(), 
			t_descp.data<float>() + t_descp.numel(), 
			m_descp.data()
		);
	}

};

#endif DESCRIPTOR_H