#ifndef ESTIMATORRES_H
#define ESTIMATORRES_H

#include "dtypes/keypoint.hpp"

struct EstimatorRes {
	Eigen::Matrix<float, 3, 3> M;
	std::vector<Keypoint> kps;
};

#endif ESTIMATORRES_H