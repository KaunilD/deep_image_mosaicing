#include "libs.hpp"
#include "keypoint.hpp"
namespace Utils {


	void compute_heatmap(cv::Mat& heatmap, const std::vector<Keypoint> points) {
		for (auto it = points.begin(); it != points.end(); it++) {
			heatmap.at<uchar>(it->row, it->col) = 255;
		};
	}

}