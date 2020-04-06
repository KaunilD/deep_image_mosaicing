// Image Mosaicing.cpp : Defines the entry point for the application.
//
#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

#include "imagemosaicing.hpp"
#include "image.hpp"

using namespace std;

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

std::vector<std::tuple<int, int, float>> nms_fast(int h, int w, std::vector<tuple<int, int, float>>& points, int nms_distance) {
	using std::vector;
	using std::tuple;

	vector<vector<int>> grid(h, vector<int>(w, 0));
	vector<tuple<int, int, float>> r_points;
	// sort as per scores

	std::sort(points.begin(), points.end(), [](tuple<int, int, float> t1, tuple<int, int, float> t2) -> bool{
		return std::get<2>(t1) > std::get<2>(t2);
		});
	
	for (int i = 0; i < points.size(); i++) {
		int row = std::get<0>(points.at(i));
		int col = std::get<1>(points.at(i));
	
		grid[row][col] = 1;
	}

	int suppressed_points = 0;

	for (int i = 0; i < points.size(); i++) {
		int row = std::get<0>(points.at(i));
		int col = std::get<1>(points.at(i));
		float val = std::get<2>(points.at(i));
		if (row > nms_distance && row < h - nms_distance && col > nms_distance && col < w - nms_distance) {
			if (grid[row][col] == 1) {

				for (int k_row = -nms_distance / 2; k_row <= nms_distance / 2; k_row++) {
					for (int k_col = -nms_distance / 2; k_col <= nms_distance / 2; k_col++) {
						grid[row + k_row][col + k_col] = 0;
					}
				}
				grid[row][col] = -1;
				suppressed_points++;

				r_points.push_back(tuple<int, int, float>(row, col, val));
			}

		}

	}
	
	return r_points;

}


std::vector<std::tuple<int, int, float>> nms_fast(torch::IntArrayRef size, torch::Tensor& points, int nms_distance) {
	using torch::Tensor;
	
	int H = size[0];
	int W = size[1];

	Tensor grid = torch::zeros(size);

	/*
		converting to an std container to sort as per scores
		will imporve upon by using tensor::sort later.
	*/
	std::vector<std::tuple<int, int, float>> std_points, r_points;
	for (int i = 0; i < points.size(1); i++) {
		int row = points[0][i].item<int>();
		int col = points[1][i].item<int>();

		auto tuple = std::tuple<int, int, float>(row, col, points[2][i].item<float>());

		std_points.push_back(tuple);
	}
	/* sort as per confidence score in descending order*/
	std::sort(std_points.begin(), std_points.end(), 
		[](tuple<int, int, float> t1, tuple<int, int, float> t2) -> bool {
			return std::get<2>(t1) > std::get<2>(t2);
		}
	);
	/*
		initialize the grid.
		1	= not visited
		-1	= selected
		0	= suppressed
	*/
	for (int i = 0; i < std_points.size(); i++) {
		int row = std::get<0>(std_points.at(i));
		int col = std::get<1>(std_points.at(i));

		grid[row][col] = 1;
	}

	int suppressed_points = 0;

	for (int i = 0; i < std_points.size(); i++) {
		int row		= std::get<0>(std_points.at(i));
		int col		= std::get<1>(std_points.at(i));
		float val	= std::get<2>(std_points.at(i));
		/* ignore border pixels */
		if (row > nms_distance && row < H - nms_distance && col > nms_distance && col < W - nms_distance) {
			
			if (grid[row][col] == 1) {

				for (int k_row = -nms_distance / 2; k_row <= nms_distance / 2; k_row++) {
					for (int k_col = -nms_distance / 2; k_col <= nms_distance / 2; k_col++) {
						grid[row + k_row][col + k_col] = 0;
					}
				}
				grid[row][col] = -1;
				suppressed_points++;

				r_points.push_back(tuple<int, int, float>(row, col, val));
			}
		}
	}

	return r_points;

}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_kp_desc(
	const torch::Tensor& img, torch::jit::script::Module& module
) {
	using torch::Tensor;

	std::vector<torch::jit::IValue> inputs;
	Tensor points;
	std::vector<std::tuple<int, int, float>> supressed_points;
	
	module.eval();
	
	inputs.push_back(img);
	auto outputs = module.forward(inputs).toTuple();

	auto semi = outputs->elements()[0].toTensor();
	auto coarse_desc = outputs->elements()[1].toTensor();

	semi = semi.squeeze();
	auto dense = torch::exp(semi);
	dense /= (torch::sum(dense, 0) + 1e-5);

	dense = dense.narrow(0, 0, dense.size(0) - 1);

	int H = static_cast<int>(img.size(0)) / 8;
	int W = static_cast<int>(img.size(1)) / 8;

	dense = dense.permute({ 1, 2, 0 });

	auto heatmap = dense.reshape({ H, W, 8, 8 });

	heatmap = heatmap.permute({0, 2, 1, 3});
	heatmap = heatmap.reshape({H*8, W*8});

	auto xy_idx = heatmap.where(
		torch::operator>=(heatmap, 0.015f), torch::zeros(1)
	).nonzero();

	xy_idx = xy_idx.transpose;
	
	points = torch::zeros(
		{3, static_cast<int>(xy_idx.size(0))}
	);
	points[0] = xy_idx[0];
	points[1] = xy_idx[1];
	points[2] = heatmap[points[0]][points[1]];

	auto size = img.sizes();
	supressed_points = nms_fast(size, heatmap, 4);
}


int main()
{
	
	Image a("C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\icl_snippet\\250.png");

	torch::Tensor a_tensor = matToTensor(a.m_image);
	a_tensor = a_tensor.permute({ 0, 3, 1, 2 });

	std::vector<torch::jit::IValue> inputs;

	inputs.push_back(a_tensor);
	
	torch::jit::script::Module module;
	try {
		module = torch::jit::load(
			"C:\\Users\\dhruv\\Development\\git\\image_mosaicing\\res\\traced_superpoint_v1.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	module.eval();

	auto outputs = module.forward(inputs).toTuple();
	
	torch::Tensor semi = outputs->elements()[0].toTensor();
	torch::Tensor desc = outputs->elements()[1].toTensor();
	std::cout << semi.sizes() << " " << desc.sizes() << "\n";
	// Keypoints
	semi = semi.squeeze();
	std::cout << semi.sizes() << "\n";
	
	torch::Tensor dense = torch::exp(semi);
	dense = dense / (torch::sum(dense, 0)+0.00001);
	std::cout << dense.sizes() << "\n";

	torch::Tensor nodust = dense.narrow(0, 0, dense.size(0)-1);
	std::cout << nodust.sizes() << "\n";
	
	int H = (int)a.m_height / 8;
	int W = (int)a.m_width / 8;

	nodust = nodust.permute({1, 2, 0});
	std::cout << nodust.sizes() << "\n";
	
	torch::Tensor heatmap = nodust.reshape({H, W, 8, 8});
	std::cout << heatmap.sizes() << "\n";

	heatmap = heatmap.permute({ 0, 2, 1, 3 });
	std::cout << heatmap.sizes() << "\n";

	heatmap = heatmap.reshape({ H*8, W*8 });
	std::cout << heatmap.sizes() << "\n";

	cv::imwrite("heatmap.png", tensorToMat(heatmap));

	torch::Tensor xy_idx = heatmap.where(torch::operator>=(heatmap, 0.015f), torch::zeros(1)).nonzero();
	std::cout << xy_idx.sizes() << "\n";

	std::vector<std::tuple<int, int, float>> points;
	
	for (int i = 0; i < xy_idx.size(0); i++) {
		int row = xy_idx[i][0].item<int>();
		int col = xy_idx[i][1].item<int>();

		auto tuple = std::tuple<int, int, float>(row, col, heatmap[row][col].item<float>());

		points.push_back(tuple);
	}

	std::vector<std::tuple<int, int, float>> s_points = nms_fast(heatmap.size(0), heatmap.size(1), points, 4);

	std::cout << points.size() << " " << s_points.size() << "\n";
	
	cv::Mat s_heatmap = cv::Mat::zeros(heatmap.size(0), heatmap.size(1), CV_8UC1);
	for (int i = 0; i < s_points.size(); i++) {
		int row = std::get<0>(s_points.at(i));
		int col = std::get<1>(s_points.at(i));
		
		s_heatmap.at<uchar>(row, col) = 255;
	}

	cv::imwrite("s_heatmap.png", s_heatmap);

	// Descriptors
	int D = desc.size(1);
	std::cout << desc.sizes() << "\n";
	
	torch::Tensor sample_pts = torch::zeros({ 2, static_cast<int>(s_points.size()) });
	for (int i = 0; i < s_points.size(); i++) {
		sample_pts[0][i] = std::get<0>(s_points.at(i));
		sample_pts[1][i] = std::get<1>(s_points.at(i));
	}
	sample_pts[0] = (sample_pts[0] / (float(W) / 2.0f)) - 1.0f;
	sample_pts[1] = (sample_pts[1] / (float(H) / 2.0f)) - 1.0f;
	sample_pts = sample_pts.permute({0, 1}).contiguous();
	sample_pts = sample_pts.view({ 1, 1, -1, 2 });
	sample_pts = sample_pts.to(torch::kF32);
	
	desc = torch::nn::functional::grid_sample(desc, sample_pts);
	desc = desc.reshape({ D, -1 });
	desc /= desc.norm();
	std::cout << desc.sizes();

	return 0;
}
