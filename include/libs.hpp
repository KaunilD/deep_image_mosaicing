#ifndef LIBS_H
#define LIBS_H


#include <iostream>
#include <memory>
#include <string>
#include <vector>

// OpenCV
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/surface_matching/icp.hpp"

// litorch
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Aten.h>
#include <c10/cuda/CUDACachingAllocator.h>

using namespace std;

using std::unique_ptr;
using std::make_unique;
using std::shared_ptr;
using std::make_shared;

#endif LIBS_H