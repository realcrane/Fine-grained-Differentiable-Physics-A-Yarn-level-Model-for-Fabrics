#pragma once
#include "Solvers.h"

Tensor sub_signed_vf_distance(const Tensor& y0, const Tensor& y1, const Tensor y2, Tensor& n, std::array<Tensor, 4>& ws, double thres, bool& over);

Tensor sub_signed_ee_distance(const Tensor& x1mx0, const Tensor& y0mx0, const Tensor& y1mx0, const Tensor& y0mx1, const Tensor& y1mx1, const Tensor& y1my0,
	Tensor& n, std::array<Tensor, 4>& ws, double thres, bool& over);
