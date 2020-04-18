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
		return t1.m_conf > t2.m_conf;
		});

	/*
		initialize grid
		-1	= kept
		0	= supressed
		1	= unvisited
	*/
	for (int i = 0; i < points->size(); i++) {
		grid[points->at(i).m_loc(0)][points->at(i).m_loc(1)] = 1;

	}

	int suppressed_points = 0;

	for (int i = 0; i < points->size(); i++) {
		int row = points->at(i).m_loc(0);
		int col = points->at(i).m_loc(1);
		float val = points->at(i).m_conf;
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

	auto dense	= torch::softmax(semi, 1);

	auto nodust = dense.slice(1, 0, 64);
	nodust = nodust.permute({ 0, 2, 3, 1 });

	int H = nodust.size(1);
	int W = nodust.size(2);

	semi = nodust.contiguous().view({-1, H, W, 8, 8});
	semi = semi.permute({0, 1, 3, 2, 4});
	auto heatmap = semi.contiguous().view({-1, H*8, W*8});
	
	heatmap = heatmap.squeeze(0);

	auto yx_idx = heatmap > 0.015f;

	yx_idx = torch::nonzero(yx_idx);

	auto rows = yx_idx.select(1, 0);
	auto cols = yx_idx.select(1, 1);

	for (int i = 0; i < yx_idx.size(0); i++) {
		int row = rows[i].item<int>();
		int col = cols[i].item<int>();

		Keypoint keypoint(
			row, col, heatmap[row][col].item<float>());

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
	auto sample_pts = torch::zeros({ static_cast<int>(keypoints->size()), 2});
	for (int i = 0; i < keypoints->size(); i++) {
		sample_pts[i][0] = keypoints->at(i).m_loc(0);
		sample_pts[i][1] = keypoints->at(i).m_loc(1);
	}
	sample_pts = sample_pts.to(torch::kFloat);

	auto grid = torch::zeros({ 1, 1, sample_pts.size(0), 2 });
	
	/*
		z-score points for grid zampler
	*/
	grid[0][0].slice(1, 0, 1) = 2.0f * sample_pts.slice(1, 1, 2) / (float(img.width) - 1.0f); // xs
	grid[0][0].slice(1, 1, 2) = 2.0f * sample_pts.slice(1, 0, 1) / (float(img.height) - 1.0f); // ys
	
	auto desc = torch::grid_sampler(pred_desc, grid, 0, 0, false);
	desc = desc.squeeze(0).squeeze(1);

	auto dn = torch::norm(desc, 2, 1);
	desc = desc.div(torch::unsqueeze(dn, 1));

	desc = desc.transpose(0, 1).contiguous();

	unique_ptr<std::vector<Descriptor>> descriptors(new std::vector<Descriptor>());
	
	for (int i = 0; i < desc.size(0); i++) {
		Descriptor descriptor(desc[i]);
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
	
	std::cout << semi.sizes() << " " << desc.sizes() << "\n";

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