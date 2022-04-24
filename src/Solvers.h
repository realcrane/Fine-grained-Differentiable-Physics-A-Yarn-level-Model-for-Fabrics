#pragma once
#include "Eigen/Eigen"
#include "Eigen/SparseCore"

#include "Constants.h"

template <typename T>
T newtows_method(T a, T b, T c, T d, T x0, int init_dir) {
    if (init_dir != 0) {
        T y0 = d + x0 * (c + x0 * (b + x0 * a)),
            ddy0 = 2 * b + (x0 + init_dir * 1e-6) * (6 * a);
        x0 += init_dir * sqrt(abs(2 * y0 / ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        T y = d + x0 * (c + x0 * (b + x0 * a));
        T dy = c + x0 * (2 * b + x0 * 3 * a);
        if (dy == 0)
            return x0;
        T x1 = x0 - y / dy;
        if (abs(x0 - x1) < 1e-6)
            return x0;
        x0 = x1;
    }
    return x0;
}

inline Tensor TorchSolver(const Tensor& A, const Tensor& b) {
    return torch::linalg::solve(A, b);
}

Tensor EigenSolverForward(const Tensor& A, const Tensor& b, bool verb=false);

std::vector<Tensor> EigenSolverBackward(Tensor& dldz, const Tensor& ans, const Tensor& tensor_A, const Tensor& tensor_b);

class EigenSolver : public torch::autograd::Function<EigenSolver> {
public:

    static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor A, Tensor b) {
        auto output = EigenSolverForward(A, b);
        ctx->save_for_backward({ output, A, b });
        return output;
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list dldz) {
        auto saved = ctx->get_saved_variables();
        auto ans = saved[0];
        auto A = saved[1];
        auto b = saved[2];
        std::vector<Tensor> ans_back = EigenSolverBackward(dldz[0], ans, A, b);
        return { ans_back[0], ans_back[1] };
    }
};

Tensor newtons_method(Tensor a, Tensor b, Tensor c, Tensor d, Tensor x0, int init_dir);

int solve_quadratic(Tensor a, Tensor b, Tensor c, Tensor& x0, Tensor& x1);

Tensor solve_cubic_forward(Tensor a, Tensor b, Tensor c, Tensor d);

std::vector<Tensor> solve_cubic_backward(Tensor dldz, Tensor ans, Tensor a, Tensor b, Tensor c, Tensor d);

class CubicSolver : public torch::autograd::Function<CubicSolver> {
public:
    
    static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor a, Tensor b, Tensor c, Tensor d) {
        auto ans = solve_cubic_forward(a, b, c, d);
        ctx->save_for_backward({ans, a, b, c, d});
        return ans;
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list dldz) {
        auto saved = ctx->get_saved_variables();
        auto ans = saved[0];
        auto a = saved[1];
        auto b = saved[2];
        auto c = saved[3];
        auto d = saved[4];
        std::vector<Tensor> ans_back = solve_cubic_backward(dldz[0], ans, a, b, c, d);
        return { ans_back[0], ans_back[1], ans_back[2], ans_back[3] };
    }
};