#ifndef FEATURES_H
#define FEATURES_H

#include "libs.hpp"
#include "keypoint.hpp"
#include "descriptor.hpp"

struct Features {
	unique_ptr<std::vector<Keypoint>>	m_keypoints;
	unique_ptr<std::vector<Descriptor>>	m_descriptors;

	Features() {
		m_keypoints		= make_unique<std::vector<Keypoint>>();
		m_descriptors	= make_unique<std::vector<Descriptor>>();
	};

	Features(unique_ptr<std::vector<Keypoint>> keypoints, unique_ptr<std::vector<Descriptor>> descriptors):
		m_keypoints(std::move(keypoints)), m_descriptors(std::move(descriptors)) {
	}
};
#endif FEATURES_H