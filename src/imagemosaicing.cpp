// Image Mosaicing.cpp : Defines the entry point for the application.

#include "utils.hpp"
#include "imagemosaicing.hpp"



int main()
{
	std::cout<<"Main" << " imread()" << std::endl;
	Image img_l("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\icl_snippet\\250.png");

	SuperPointExtractor superPointExtractor("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\traced_superpoint_v1.pt");
	shared_ptr<std::vector<unique_ptr<Features>>> features = superPointExtractor.run({img_l});

	cv::Mat heatmap = cv::Mat::zeros(1000, 1000, CV_8UC1);
	Utils::compute_heatmap(heatmap, *(features->at(0)->m_keypoints).get());
	cv::imwrite("heatmap.png", heatmap);
	return 0;
}
