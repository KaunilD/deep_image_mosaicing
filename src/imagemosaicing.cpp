// Image Mosaicing.cpp : Defines the entry point for the application.

#include "utils.hpp"
#include "imagemosaicing.hpp"

/*
	temp code
	cv::Mat heatmap = cv::Mat::zeros(1000, 1000, CV_8UC1);
	Utils::compute_heatmap(heatmap, *(features->at(0)->m_keypoints).get());
	cv::imwrite("heatmap.png", heatmap);

*/

int main()
{
	std::cout<<"Main" << " imread()" << std::endl;
	Image img_l("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\icl_snippet\\250.png");
	Image img_r("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\icl_snippet\\254.png");

	SuperPointExtractor superPointExtractor("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\traced_superpoint_v1.pt");
	shared_ptr<std::vector<shared_ptr<Features>>> features = superPointExtractor.run({img_l, img_r});

	BFMatcher bfmatcher;

	std::vector<MatchPair> matches = bfmatcher.run(
		*features->at(0)->m_descriptors.get(), *features->at(1)->m_descriptors.get());

	std::cout << matches.size() << "\n";
	return 0;
}
