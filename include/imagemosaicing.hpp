#ifndef IMGMOS_H
#define IMGMOS_H

#include "libs.hpp"

#include "image.hpp"
#include "keypoint.hpp"
#include "descriptor.hpp"

#include "superpointextractor.hpp"

class ImageMosaic {
public:
	unique_ptr<std::vector<Image>> m_images;
	ImageMosaic() {
		m_images = make_unique<std::vector<Image>>();
	};
	void addImage(const Image& t_image) {
		assert(m_images);
		m_images->push_back(t_image);
	}
};

#endif IMGMOS_H