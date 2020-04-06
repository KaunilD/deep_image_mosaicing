// Image Mosaicing.cpp : Defines the entry point for the application.
//
#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

#include "imagemosaicing.hpp"
#include "image.hpp"

using namespace std;

struct Keypoint {
	int row, col;
	float conf;

	Keypoint(int r, int c, int v) : row(r), col(c), conf(v) {};
};

torch::Tensor matToTensor(const cv::Mat& image) {
	std::vector<int64_t> dims = { 1, image.rows, image.cols, image.channels() };
	return torch::from_blob(
		image.data, dims, torch::kFloat
	);
}

cv::Mat tensorToMat(const torch::Tensor& tensor) {
	assert(tensor.sizes().size() == 2); /*only 2D tensors supported*/

	auto temp = tensor.clone();
	temp = temp.mul(255).clamp(0, 255).to(torch::kU8);

	cv::Mat mat(tensor.size(0), tensor.size(1), CV_8UC1);
	std::memcpy(mat.data, temp.data_ptr(), sizeof(torch::kU8)*tensor.numel());
	return mat;
}

std::vector<Keypoint> nms_fast(
	int h, int w, std::vector<Keypoint>& points, int nms_distance
) {
	using std::vector;
	using std::tuple;

	vector<vector<int>> grid(h, vector<int>(w, 0));
	vector<Keypoint> r_points;
	
	// sort as per scores
	std::sort(points.begin(), points.end(), [](Keypoint t1, Keypoint t2) -> bool{
		return t1.conf > t2.conf;
		});
	
	/*
		initialize grid
		-1	= kept
		0	= supressed
		1	= unvisited
	*/
	for (int i = 0; i < points.size(); i++) {
		grid[points[i].row][points[i].col] = 1;
		
	}

	int suppressed_points = 0;

	for (int i = 0; i < points.size(); i++) {
		int row		= points[i].row;
		int col		= points[i].col;
		float val	= points[i].conf;
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

				r_points.push_back(Keypoint(row, col, val));
			}

		}

	}
	
	return r_points;

}



std::vector<Keypoint> compute_kp(const cv::Size& img, torch::Tensor& semi ) {
	
	std::vector<Keypoint> points, supressed_points;

	semi = semi.squeeze();
	auto dense = torch::exp(semi);
	dense /= (torch::sum(dense, 0) + 1e-5);

	auto nodust = dense.narrow(0, 0, dense.size(0) - 1);
	std::cout << nodust.sizes() << "\n";

	int H = static_cast<int>(img.height) / 8;
	int W = static_cast<int>(img.width) / 8;

	nodust = nodust.permute({ 1, 2, 0 });
	std::cout << nodust.sizes() << "\n";

	auto heatmap = nodust.reshape({ H, W, 8, 8 });

	heatmap = heatmap.permute({ 0, 2, 1, 3 });
	heatmap = heatmap.reshape({ H * 8, W * 8 });

	auto xy_idx = heatmap.where(
		torch::operator>=(heatmap, 0.015f), torch::zeros(1)
		).nonzero();

	for (int i = 0; i < xy_idx.size(0); i++) {
		int row = xy_idx[i][0].item<int>();
		int col = xy_idx[i][1].item<int>();

		auto tuple = Keypoint(
			row, col, heatmap[row][col].item<float>());

		points.push_back(tuple);
	}

	supressed_points = nms_fast(heatmap.size(0), heatmap.size(1), points, 4);

	cv::Mat s_heatmap = cv::Mat::zeros(heatmap.size(0), heatmap.size(1), CV_8UC1);

	for (int i = 0; i < supressed_points.size(); i++) {
		int row = supressed_points.at(i).row;
		int col = supressed_points.at(i).col;

		s_heatmap.at<uchar>(row, col) = 255;
	}
	cv::imwrite("s_heatmap_nms.png", s_heatmap);
	
	return supressed_points;
}

torch::jit::script::Module load_module(const std::string& path) {

	torch::jit::script::Module module;
	try {
		module = torch::jit::load(path);
	}
	catch (const c10::Error & e) {
		std::cerr << "error loading the model\n";
	}

	return module;
}



int main()
{
	
	cv::Mat image = cv::imread("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\icl_snippet\\250.png", 0);
	image.convertTo(image, CV_32FC1, 1.0 / 255.0f, 0);


	torch::Tensor a_tensor = matToTensor(image);
	a_tensor = a_tensor.permute({ 0, 3, 1, 2 });

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(a_tensor);

	torch::jit::script::Module module = load_module(
		"C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\traced_superpoint_v1.pt"
	);
	module.eval();

	auto outputs = module.forward(inputs).toTuple();
	
	torch::Tensor semi = outputs->elements()[0].toTensor();
	torch::Tensor desc = outputs->elements()[1].toTensor();
	
	std::vector<Keypoint> supressed_points = compute_kp(image.size(), semi);


	// Descriptors
	int D = desc.size(1);
	std::cout << desc.sizes() << "\n";
	
	auto sample_pts = torch::zeros({ 2, static_cast<int>(supressed_points.size()) });
	for (int i = 0; i < supressed_points.size(); i++) {
		sample_pts[0][i] = supressed_points.at(i).row;
		sample_pts[1][i] = supressed_points.at(i).col;
	}
	/*
		z-score points for grid zampler
	*/
	sample_pts[0] = (sample_pts[0] / (float(image.rows) / 2.0f)) - 1.0f;
	sample_pts[1] = (sample_pts[1] / (float(image.cols) / 2.0f)) - 1.0f;
	sample_pts = sample_pts.permute({0, 1}).contiguous();
	sample_pts = sample_pts.view({ 1, 1, -1, 2 });
	sample_pts = sample_pts.to(torch::kF32);
	
	desc = torch::nn::functional::grid_sample(desc, sample_pts);
	desc = desc.reshape({ D, -1 });
	desc /= desc.norm();
	
	return 0;
}
