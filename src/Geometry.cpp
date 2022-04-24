#include "Geometry.h"

inline Tensor norm2(Tensor x) {
    return torch::dot(x.squeeze(), x.squeeze());
}

inline Tensor normalized(Tensor x) {
    Tensor l{ torch::norm(x) };
    if (l.item<double>() == 0.0) {
        return x;
    }
    return x / l;
}

Tensor sub_signed_vf_distance(const Tensor& y0, const Tensor& y1, const Tensor y2, Tensor& n, std::array<Tensor, 4>& ws, double thres, bool& over) {

    n = cross(y1 - y0, y2 - y0);
    ws[0] = torch::zeros({ 1 }, opts);
    ws[1] = torch::zeros({ 1 }, opts);
    ws[2] = torch::zeros({ 1 }, opts);
    ws[3] = torch::zeros({ 1 }, opts);

    if (norm2(n).item<double>() < thres) {
        over = true;
        return torch::tensor({std::numeric_limits<double>::infinity()}, opts);
    }
    n = normalized(n);
    Tensor h{ -torch::dot(y0.squeeze(), n.squeeze()) };
    if (torch::abs(h).item<double>() > thres) {
        over = true;
        return h;
    }
    else {
        over = false;
    }

	Tensor b0{ torch::mm(y1.transpose(0,1), torch::cross(y2, n)).squeeze() };
    Tensor b1{ torch::mm(y2.transpose(0,1), torch::cross(y0, n)).squeeze() };
    Tensor b2{ torch::mm(y0.transpose(0,1), torch::cross(y1, n)).squeeze() };
    Tensor sum{ 1.0 / (b0 + b1 + b2) };
	ws[0] = torch::ones({ 1 }, opts);
    ws[1] = -b0 * sum;
    ws[2] = -b1 * sum;
    ws[3] = -b2 * sum;
    return h;
}

Tensor sub_signed_ee_distance(const Tensor& x1mx0, const Tensor& y0mx0, const Tensor& y1mx0, const Tensor& y0mx1, const Tensor& y1mx1, const Tensor& y1my0,
	Tensor& n, std::array<Tensor, 4>& ws, double thres, bool& over) {

    n = cross(x1mx0, y1my0);
    ws[0] = torch::zeros({ 1 }, opts);
    ws[1] = torch::zeros({ 1 }, opts);
    ws[2] = torch::zeros({ 1 }, opts);
    ws[3] = torch::zeros({ 1 }, opts);

    if (norm2(n).item<double>() < thres) {
        over = true;
        return torch::tensor({ std::numeric_limits<double>::infinity() }, opts);
    }

    n = normalized(n);
    Tensor h{ -torch::dot(y0mx0.squeeze(), n.squeeze()) };
    if (torch::abs(h).item<double>() > thres) {
        over = true;
        return h;
    }
    else {
        over = false;
    }

    Tensor a0{ torch::mm(y1mx1.transpose(0,1), torch::cross(y0mx1, n)).squeeze() };
    Tensor a1{ torch::mm(y0mx0.transpose(0,1), torch::cross(y1mx0, n)).squeeze() };
    Tensor b0{ torch::mm(y1mx1.transpose(0,1), torch::cross(y1mx0, n)).squeeze() };
    Tensor b1{ torch::mm(y0mx0.transpose(0,1), torch::cross(y0mx1, n)).squeeze() };
    Tensor suma{ 1.0 / (a0 + a1) };
    Tensor sumb{ 1.0 / (b0 + b1) };
    ws[0] = a0 * suma;
    ws[1] = a1 * suma;
    ws[2] = -b0 * sumb;
    ws[3] = -b1 * sumb;

    return h;
}