// Image Mosaicing.cpp : Defines the entry point for the application.
//
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Aten.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/surface_matching/icp.hpp"

#include <iostream>
#include <memory>

#include "imagemosaicing.hpp"
#include "image.hpp"

#define LOG2(x, y) {std::cout << x << "::" << y <<"\n";}
#define LOG(x) {std::cout << x << "\n";}
using namespace std;

using std::unique_ptr;
using std::make_unique;
using std::shared_ptr;
using std::make_shared;

struct Keypoint {
	int row, col;
	float conf;

	Keypoint(int r, int c, int v) : row(r), col(c), conf(v) {};
	
};

struct Features{
	shared_ptr<std::vector<Keypoint>> m_keypoints;
	shared_ptr<torch::Tensor> m_descriptors;

	Features(shared_ptr<std::vector<Keypoint>> keypoints, shared_ptr<torch::Tensor> descriptors) {
		m_keypoints = keypoints;
		m_descriptors = descriptors;
	}
};

torch::Tensor matToTensor(const cv::Mat& image) {
	std::vector<int64_t> dims = { 1, image.rows, image.cols, image.channels() };
	return torch::from_blob(
		image.data, dims, torch::kFloat
	);
}

cv::Mat tensorToMat(const torch::Tensor& tensor) {
	assert(tensor.sizes().size() == 2); /*only 2D tensors supported*/

	auto temp	= tensor.clone();
	temp		= temp.mul(255).clamp(0, 255).to(torch::kU8);

	cv::Mat mat(tensor.size(0), tensor.size(1), CV_8UC1);
	std::memcpy(mat.data, temp.data_ptr(), sizeof(torch::kU8)*tensor.numel());
	return mat;
}

std::unique_ptr<std::vector<Keypoint>> nms_fast(
	int h, int w, std::unique_ptr<std::vector<Keypoint>> points, int nms_distance
) {

	LOG("nms_fast");

	using std::vector;
	using std::tuple;

	vector<vector<int>> grid(h, vector<int>(w, 0));
	std::unique_ptr<vector<Keypoint>> r_points = make_unique<vector<Keypoint>>();
	
	// sort as per scores
	std::sort(points->begin(), points->end(), [](Keypoint t1, Keypoint t2) -> bool{
		return t1.conf > t2.conf;
		});
	
	/*
		initialize grid
		-1	= kept
		0	= supressed
		1	= unvisited
	*/
	for (int i = 0; i < points->size(); i++) {
		grid[points->at(i).row][points->at(i).col] = 1;
		
	}

	int suppressed_points = 0;

	for (int i = 0; i < points->size(); i++) {
		int row		= points->at(i).row;
		int col		= points->at(i).col;
		float val	= points->at(i).conf;
		/*
			supress border points by default
		*/
		if (row > nms_distance && row < h - nms_distance && col > nms_distance && col < w - nms_distance) {
			if (grid[row][col] == 1) {

				for (int k_row = -nms_distance / 2; k_row <= nms_distance / 2; k_row++) {
					for (int k_col = -nms_distance / 2; k_col <= nms_distance / 2; k_col++) {
						grid[row + k_row][col + k_col] = 0;
					}
				}
				grid[row][col] = -1;
				suppressed_points++;

				r_points->push_back(Keypoint(row, col, val));
			}

		}

	}
	
	return std::move(r_points);

}


std::unique_ptr<std::vector<Keypoint>> compute_keypoints(const cv::Size& img, torch::Tensor& semi ) {
	LOG("compute_keyoints");
	std::unique_ptr<std::vector<Keypoint>> points = make_unique<std::vector<Keypoint>>();
	std::unique_ptr<std::vector<Keypoint>> supressed_points;

	semi = semi.squeeze();

	auto dense	= torch::exp(semi);
	dense		= dense/(torch::sum(dense, 0) + 1e-5);

	auto nodust = dense.narrow(0, 0, dense.size(0) - 1);
	
	int H = static_cast<int>(img.height) / 8;
	int W = static_cast<int>(img.width) / 8;

	nodust = nodust.permute({ 1, 2, 0 });
	
	auto heatmap = nodust.reshape({ H, W, 8, 8 });

	heatmap = heatmap.permute({ 0, 2, 1, 3 });
	heatmap = heatmap.reshape({ H * 8, W * 8 });

	auto xy_idx = heatmap.where(
		torch::operator>=(heatmap, 0.015f), torch::zeros(1)
		).nonzero();

	for (int i = 0; i < xy_idx.size(0); i++) {
		int row = xy_idx[i][0].item<int>();
		int col = xy_idx[i][1].item<int>();

		Keypoint keypoint(
			row, col, heatmap[row][col].item<float>());

		points->push_back(std::move(keypoint));
	}

	supressed_points = nms_fast(heatmap.size(0), heatmap.size(1), std::move(points), 4);

	return std::move(supressed_points);
}

shared_ptr<torch::jit::script::Module> load_module(const std::string& path, torch::Device device) {
	LOG("load_module::reading model");
	shared_ptr<torch::jit::script::Module> module;
	try {
		module = make_shared<torch::jit::script::Module>(torch::jit::load(path, device));
	}
	catch (const c10::Error & e) {
		std::cerr << "error loading the model\n";
	}
	LOG("load_module::model loaded");
	return std::move(module);
}

torch::DeviceType get_device() {
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	return device_type;
}

torch::Tensor prepare_tensor(const torch::Tensor& tensor, const torch::Device& device) {
	torch::Tensor a = tensor.permute({ 0, 3, 1, 2 });
	a = a.to(device);
	return std::move(a);
}


torch::Tensor compute_descriptors(
	const torch::Tensor& pred_desc, const vector<Keypoint>* keypoints, const cv::Size& img
) {
	LOG("compute_descriptors");
	// Descriptors
	int D = pred_desc.size(1);
	auto sample_pts = torch::zeros({ 2, static_cast<int>(keypoints->size()) });
	for (int i = 0; i < keypoints->size(); i++) {
		sample_pts[0][i] = keypoints->at(i).row;
		sample_pts[1][i] = keypoints->at(i).col;
	}
	/*
		z-score points for grid zampler
	*/
	sample_pts[0] = (sample_pts[0] / (float(img.height) / 2.0f)) - 1.0f;
	sample_pts[1] = (sample_pts[1] / (float(img.width) / 2.0f)) - 1.0f;
	
	sample_pts = sample_pts.permute({ 0, 1 }).contiguous();
	sample_pts = sample_pts.view({ 1, 1, -1, 2 });
	sample_pts = sample_pts.to(torch::kF32);

	auto desc = torch::nn::functional::grid_sample(pred_desc, sample_pts);
	desc		= desc.reshape({ D, -1 });
	desc		= desc / desc.norm();

	return std::move(desc);
}

unique_ptr<Features> compute_features(const Image& img, shared_ptr<torch::jit::script::Module> module, const torch::Device& device) {
	module->eval();

	torch::Tensor img_tensor = matToTensor(img.m_image);
	img_tensor = prepare_tensor(img_tensor, device);

	LOG2("Main", img_tensor.sizes());

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);

	LOG("Main::forawrd");
	
	auto outputs = module->forward(inputs).toTuple();

	torch::Tensor semi = outputs->elements()[0].toTensor().to(torch::kCPU);
	torch::Tensor desc = outputs->elements()[1].toTensor().to(torch::kCPU);
	LOG2(semi.sizes(), desc.sizes());

	std::unique_ptr<std::vector<Keypoint>> keypoints = compute_keypoints(img.m_image.size(), semi);
	std::unique_ptr<torch::Tensor> descriptors = make_unique<torch::Tensor>(compute_descriptors(desc, keypoints.get(), img.m_image.size()));

	c10::cuda::CUDACachingAllocator::emptyCache();

	return std::move(make_unique<Features>(std::move(keypoints), std::move(descriptors)));
}


void compute_heatmap(cv::Mat& heatmap, const shared_ptr<std::vector<Keypoint>> points) {
	for (auto it = points->begin(); it != points->end(); it++) {
		heatmap.at<uchar>(it->row, it->col) = 255;
	};
}


int main()
{
	torch::Device device(get_device());
	
	shared_ptr<torch::jit::script::Module> module = load_module(
		"C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\traced_superpoint_v1.pt", device);

	LOG2("Main", "imread()");
	Image img_l("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\tma\\franklins_WV02.tif");
	Image img_r("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\tma\\CA228132V0108_CROPPED_2.tif");

	unique_ptr<Features> features_l = compute_features(img_l, module, device);
	unique_ptr<Features> features_r = compute_features(img_r, module, device);

	LOG2(features_l->m_keypoints->size(), features_l->m_descriptors->sizes());
	LOG2(features_r->m_keypoints->size(), features_r->m_descriptors->sizes());

	cv::Mat heatmap_l = cv::Mat::zeros(1000, 1000, CV_8UC1);
	cv::Mat heatmap_r = cv::Mat::zeros(1000, 1000, CV_8UC1);


	return 0;
}
