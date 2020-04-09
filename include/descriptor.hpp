#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include "libs.hpp"

struct Descriptor {
	using Vec2df	= std::vector<std::vector<float>>;
	
	unique_ptr<Vec2df> m_descp;
	
	int id;

	Descriptor() = default;

	Descriptor(Vec2df t_descp) {
		m_descp = make_unique<Vec2df>(t_descp.size(), std::vector<float>(t_descp.size(), 0.0f));
	}

};

#endif DESCRIPTOR_H