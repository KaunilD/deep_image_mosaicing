// Image Mosaicing.cpp : Defines the entry point for the application.

#include "utils.hpp"
#include "imagemosaicing.hpp"

/*
	temp code
	Draw matches
	for (const auto& match: matches) {
		auto kp1 = features->at(0)->m_keypoints->at(match.kp1);
		auto kp2 = features->at(1)->m_keypoints->at(match.kp2);

		cv::line(
			matchmap,
			{ kp1.row, kp1.col },
			{ kp2.row, kp2.col },
			cv::Scalar(255, 255, 255)
		);
	}
	cv::imwrite("matchmap.png", matchmap);

	
*/

int main()
{
	std::cout<<"Main" << " imread()" << std::endl;
	Image img_l("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\icl_snippet\\250.png");
	Image img_r("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\icl_snippet\\254.png");
	
	cv::Mat matchmap;
	cv::hconcat(img_l.m_image_og, img_r.m_image_og, matchmap);
	
	SuperPointExtractor superPointExtractor("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\traced_superpoint_v1.pt");
	shared_ptr<std::vector<shared_ptr<Features>>> features = superPointExtractor.run({img_l, img_r});


	cv::Mat heatmap = cv::Mat::zeros(1000, 1000, CV_8UC1);
	Utils::compute_heatmap(heatmap, *(features->at(0)->m_keypoints).get());
	cv::imwrite("heatmap_l.png", heatmap);

	heatmap = cv::Mat::zeros(1000, 1000, CV_8UC1);
	Utils::compute_heatmap(heatmap, *(features->at(1)->m_keypoints).get());
	cv::imwrite("heatmap_r.png", heatmap);


	BFMatcher bfmatcher(true);
	std::vector<MatchPair> matches = bfmatcher.run(
		*features->at(0)->m_descriptors.get(), *features->at(1)->m_descriptors.get());
	std::cout << matches.size() << "\n";

	Estimator estimator;
	estimator.estimateEightPoints(*(features->at(0)->m_keypoints).get(), *(features->at(1)->m_keypoints).get(), matches);

	return 0;
}
