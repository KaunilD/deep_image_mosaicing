#ifndef SUPERPOINTEXT_H
#define SUPERPOINTEXT_H

#include "libs.hpp"
#include "image.hpp"
#include "dtypes/features.hpp"
#include "extractor/featureextractor.hpp"

class SuperPointExtractor: public FeatureExtractor {
public:
	
	shared_ptr<torch::jit::script::Module> model;
	torch::Device device = torch::kCPU;

	SuperPointExtractor() : FeatureExtractor() {};
	SuperPointExtractor(const std::string& modelPath);

	torch::DeviceType getDevice();
	shared_ptr<torch::jit::script::Module> loadModel(const std::string& path, torch::Device device);
	unique_ptr<std::vector<shared_ptr<Features>>> run(const std::vector<Image>& images);
	unique_ptr<std::vector<Keypoint>> nms_fast(int h, int w, unique_ptr<std::vector<Keypoint>> points, int nms_distance);
	unique_ptr<std::vector<Keypoint>> computeKeypoints(const cv::Size& img, torch::Tensor& semi);
	unique_ptr<std::vector<Descriptor>> computeDescriptors(const torch::Tensor& pred_desc, const vector<Keypoint>* keypoints, const cv::Size& img);
	unique_ptr<Features> computeFeatures(const Image& image);

	torch::Tensor matToTensor(const cv::Mat& image);
	torch::Tensor prepareTensor(const torch::Tensor& tensor, const torch::Device& device);


	void init() {

	}
};

#endif SUPERPOINTEXT_H