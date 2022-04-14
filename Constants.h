#pragma once
#include <limits>
#include "torch/torch.h"

using torch::Tensor;

const torch::Device device(torch::kCPU);

const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat64).device(device);

const Tensor U_OneHot = torch::tensor({ 1.0,0.0 }, opts).view({ 2,1 });
const Tensor V_OneHot = torch::tensor({ 0.0,1.0 }, opts).view({ 2,1 });

const Tensor EYE3 = torch::eye(3, opts);
const Tensor ZERO33 = torch::zeros({ 3,3 }, opts);
const Tensor ZERO31 = torch::zeros({ 3,1 }, opts);
const Tensor ZERO13 = torch::zeros({ 1,3 }, opts);
const Tensor ZERO11 = torch::zeros({ 1,1 }, opts);
const Tensor ZERO21 = torch::zeros({ 2,1 }, opts);
const Tensor ZERO1 = torch::zeros({ 1 }, opts);
const Tensor ONE1 = torch::ones({ 1 }, opts);
const Tensor INF = ONE1 * std::numeric_limits<double>::infinity();

static const double PI{ 3.1415926535897 };
static const double EPS{ std::numeric_limits<double>::min() };
static const int NThrd{ 8 };

