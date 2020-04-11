#include "extractor/superpointextractor.hpp"

SuperPointExtractor::SuperPointExtractor(const std::string& modelPath) {
	std::cout << "SuperPointExtractor:: " << modelPath << "\n";
	device = getDevice();
	model = loadModel(modelPath, device);
};

torch::DeviceType
SuperPointExtractor::getDevice() {
	std::cout << "SuperPointExtractor:: " << "getDevice()" << "\n";

	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "SuperPointExtractor:: " << "CUDA available! Training on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "SuperPointExtractor:: " << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	return device_type;
};

shared_ptr<torch::jit::script::Module>
SuperPointExtractor::loadModel(const std::string& path, torch::Device device) {
	std::cout << "SuperPointExtractor:: " << "loadModel" << "\n";
	shared_ptr<torch::jit::script::Module> module;
	try {
		module = make_shared<torch::jit::script::Module>(torch::jit::load(path, device));
	}
	catch (const c10::Error& e) {
		std::cerr << "SuperPointExtractor:: " << "error loading the model\n";
	}
	return std::move(module);
};

unique_ptr<std::vector<shared_ptr<Features>>>
SuperPointExtractor::run(const std::vector<Image>& images) {
	std::cout << "SuperPointExtractor:: " << "run" << "\n";
	auto features = make_unique<std::vector<shared_ptr<Features>>>();
	for (const auto& ref : images) {
		features->push_back(std::move(computeFeatures(ref)));
	}

	return std::move(features);
};

unique_ptr<std::vector<Keypoint>>
SuperPointExtractor::nms_fast(int h, int w, unique_ptr<std::vector<Keypoint>> points, int nms_distance
) {
	using std::vector;
	using std::tuple;

	vector<vector<int>> grid(h, vector<int>(w, 0));
	auto r_points = make_unique<vector<Keypoint>>();

	// sort as per scores
	std::sort(points->begin(), points->end(), [](Keypoint t1, Keypoint t2) -> bool {
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
		int row = points->at(i).row;
		int col = points->at(i).col;
		float val = points->at(i).conf;
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
	return r_points;
};

unique_ptr<std::vector<Keypoint>>
SuperPointExtractor::computeKeypoints(const cv::Size& img, torch::Tensor& semi) {
	auto points = make_unique<std::vector<Keypoint>>();
	auto supressed_points = make_unique<std::vector<Keypoint>>();

	semi = semi.squeeze();

	auto dense = torch::exp(semi);
	dense = dense / (torch::sum(dense, 0) + 1e-5);

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

	auto rows = xy_idx.select(1, 0);
	auto cols = xy_idx.select(1, 1);

	auto values = torch::index(heatmap, { rows, cols });

	for (int i = 0; i < xy_idx.size(0); i++) {
		int row = rows[i].item<int>();
		int col = cols[i].item<int>();

		Keypoint keypoint(
			row, col, values[i].item<float>());

		points->push_back(std::move(keypoint));
	}

	supressed_points = nms_fast(heatmap.size(0), heatmap.size(1), std::move(points), 4);
	return supressed_points;
};

unique_ptr<std::vector<Descriptor>>
SuperPointExtractor::computeDescriptors(
	const torch::Tensor& pred_desc, const vector<Keypoint>* keypoints, const cv::Size& img
) {
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
	desc = desc.reshape({ D, -1 });
	desc /= desc.norm(); // 256xcols


	unique_ptr<std::vector<Descriptor>> descriptors(new std::vector<Descriptor>());
	for (int i = 0; i < desc.size(0); i++) {
		Descriptor descriptor(desc[0]);
		descriptors->push_back(std::move(descriptor));
	}

	return descriptors;
};

unique_ptr<Features>
SuperPointExtractor::computeFeatures(const Image& image) {
	model->eval();

	auto i_tensor = matToTensor(image.m_image);
	i_tensor = prepareTensor(i_tensor, device);


	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(i_tensor);


	auto outputs = model->forward(inputs).toTuple();

	auto semi = outputs->elements()[0].toTensor().to(torch::kCPU);
	auto desc = outputs->elements()[1].toTensor().to(torch::kCPU);

	unique_ptr<std::vector<Keypoint>> keypoints = computeKeypoints(image.m_image.size(), semi);
	unique_ptr<std::vector<Descriptor>> descriptors = computeDescriptors(desc, keypoints.get(), image.m_image.size());

	auto f = make_unique<Features>(std::move(keypoints), std::move(descriptors));

	return f;
};

torch::Tensor
SuperPointExtractor::matToTensor(const cv::Mat& image) {
	std::vector<int64_t> dims = { 1, image.rows, image.cols, image.channels() };
	return torch::from_blob(
		image.data, dims, torch::kFloat
	);
};

torch::Tensor
SuperPointExtractor::prepareTensor(const torch::Tensor& tensor, const torch::Device& device) {
	torch::Tensor a = tensor.permute({ 0, 3, 1, 2 });
	a = a.to(device);
	return std::move(a);
};