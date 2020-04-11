#ifndef IMGMOS_H
#define IMGMOS_H

#include "libs.hpp"

#include "image.hpp"
#include "dtypes/keypoint.hpp"
#include "dtypes/descriptor.hpp"

#include "extractor/superpointextractor.hpp"
#include "matcher/bfmatcher.hpp"

/*
	WIP
*/

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