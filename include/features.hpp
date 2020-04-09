#ifndef FEATURES_H
#define FEATURES_H

#include "libs.hpp"
#include "keypoint.hpp"
#include "descriptor.hpp"

struct Features {
	using Kp_vec	= unique_ptr<std::vector<Keypoint>>;
	using Desc_vec	= unique_ptr<Descriptor>;

	Kp_vec		m_keypoints;
	Desc_vec	m_descriptors;

	Features() {
		m_keypoints		= make_unique<std::vector<Keypoint>>();
		m_descriptors	= make_unique<Descriptor>();
	};

	Features(Kp_vec keypoints, Desc_vec descriptors): 
		m_keypoints(std::move(keypoints)), m_descriptors(std::move(descriptors)) {
	}
};
#endif FEATURES_H