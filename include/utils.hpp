#include "libs.hpp"
#include "dtypes/keypoint.hpp"
namespace Utils {


	void compute_heatmap(cv::Mat& heatmap, const std::vector<Keypoint> points) {
		for (auto it = points.begin(); it != points.end(); it++) {
			heatmap.at<uchar>(it->m_loc(0), it->m_loc(1)) = 255;
		};
	}

}