#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include "libs.hpp"

struct Descriptor {
	MatrixXf_RM m_descp;
	
	int id;

	Descriptor() = default;

	Descriptor(torch::Tensor t_descp) {
		m_descp = MatrixXf_RM(
			static_cast<int>(t_descp.size(0)), static_cast<int>(t_descp.size(1)));
		
		std::copy(t_descp.data<float>(), t_descp.data<float>() + t_descp.numel(), m_descp.data());

	}

};

#endif DESCRIPTOR_H