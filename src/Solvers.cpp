#include "Solvers.h"

Tensor EigenSolverForward(const Tensor& A, const Tensor& b, bool verb) {

    // Tensor A to Eigen Sparse Matrix SpaA
    Eigen::SparseMatrix<double> SpaA(A.size(0), A.size(1));
    SpaA.reserve(Eigen::VectorXi::Constant(A.size(1), 16));
    auto A_acc = A.accessor<double, 2>();
    for (int i = 0; i < A.size(0); ++i)
        for (int j = 0; j < A.size(1); ++j)
            if (A_acc[i][j] != 0.0) {
                SpaA.insert(i, j) = A_acc[i][j];
            }
    // Tensor vector b to Eigen vector Spab
    const double* b_data = b.data_ptr<double>();
    Eigen::Map<const Eigen::VectorXd> Spab(b_data, b.size(0), b.size(1));
    // Tensor vector new velocity to Eigen vector x
    Tensor new_vel{ torch::zeros_like(b, opts) };
    double* x_data = new_vel.data_ptr<double>();
    Eigen::Map<Eigen::VectorXd> x(x_data, new_vel.size(0), new_vel.size(1));
    // Eigen CG solve  
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(SpaA);
    x = cg.solve(Spab);   
    if (verb) {
        std::cout << "#iterations:     " << cg.iterations() << std::endl;
        std::cout << "estimated error: " << cg.error() << std::endl;
        std::cout << x << std::endl;
    }
    return new_vel;
}

std::vector<Tensor> EigenSolverBackward(Tensor& dldz, const Tensor& ans, const Tensor& tensor_A, const Tensor& tensor_b) {

    // Tensor tensor_A to Eigen Sparse Matrix SpaA
    Eigen::SparseMatrix<double> SpaA(tensor_A.size(0), tensor_A.size(1));
    for (int i = 0; i < tensor_A.size(0); ++i)
        for (int j = 0; j < tensor_A.size(1); ++j)
            if (tensor_A.index({ i,j }).item<double>() != 0.0) {
                SpaA.insert(i, j) = tensor_A.index({ i,j }).item<double>();
            }
    // Tensor vector tensor_b to Eigen vector Spab
    const double* b_data = dldz.contiguous().data_ptr<double>();
    //const double* b_data = dldz.data_ptr<double>();
    Eigen::Map<const Eigen::VectorXd> Spab(b_data, tensor_b.size(0), tensor_b.size(1));
    // Tensor vector dx to Eigen vector x (recieve result)
    Tensor dx{ torch::zeros_like(tensor_b, opts) };
    double* x_data = dx.data_ptr<double>();
    Eigen::Map<Eigen::VectorXd> x(x_data, dx.size(0), dx.size(1));
    // Eigen CG solve
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(SpaA);
    x = cg.solve(Spab);

    Tensor dlda{-torch::ger(dx.squeeze(), ans.squeeze())};
    
    return { dlda, dx };
}

int solve_quadratic(Tensor a, Tensor b, Tensor c, Tensor& x0, Tensor& x1) {
	Tensor d{ torch::pow(b,2) - 4.0 * a * c };
	if (d.item<double>() < 0.0) {
		// No Solution
		x0 = -b / (2.0 * a);
		return 0;
	}
	Tensor q{ -(b + torch::sqrt(d)) / 2.0 };
	Tensor q1{ -(b - torch::sqrt(d)) / 2.0 };

	if (torch::abs(a).item<double>() > 1e-12) {
		if ((q / a).item<double>() < (q1 / a).item<double>()) {
			x0 = q / a;
			x1 = q1 / a;
		}
		else {
			x0 = q1 / a;
			x1 = q / a;
		}
		return 2;
	}
	else {
		x0 = -c / b;
		return 1;
	}
}

Tensor newtons_method(Tensor a, Tensor b, Tensor c, Tensor d, Tensor x0, int init_dir) {
    if (init_dir != 0) {
        // quadratic approximation around x0, assuming y' = 0
        Tensor  y0 = d + x0 * (c + x0 * (b + x0 * a)),
            ddy0 = 2 * b + (x0 + init_dir * 1e-6) * (6 * a);
        x0 = x0 + init_dir * torch::sqrt(torch::abs(2 * y0 / ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        Tensor y = d + x0 * (c + x0 * (b + x0 * a));
        Tensor dy = c + x0 * (2 * b + x0 * 3 * a);
        if (dy.item<double>() == 0.0)
            return x0;
        Tensor x1 = x0 - y / dy;
        if (torch::abs(x0 - x1).item<double>() < 1e-6)
            return x0;
        x0 = x1;
    }
    return x0;
}

Tensor solve_cubic_forward(Tensor a, Tensor b, Tensor c, Tensor d) {

	Tensor xc0{ torch::tensor({-1.0}, opts) },
		xc1{ torch::tensor({-1.0}, opts) };

	Tensor x0{ torch::tensor({-1.0}, opts) },
		x1{ torch::tensor({-1.0}, opts) },
		x2{ torch::tensor({-1.0}, opts) };

    int ncrit = solve_quadratic(3.0*a, 2.0*b, c, xc0, xc1);
    if (ncrit == 0) {
        return newtons_method(a, b, c, d, xc0, 0);
    }
    else if (ncrit == 1) {
        int num_solutions = solve_quadratic(b, c, d, x0, x1);
        switch (num_solutions)
        {
        case 0:
            return torch::tensor({}, opts);
        case 1:
            return x0;
        case 2:
            return torch::cat({ x0, x1 }, 0);
        default:
            return torch::tensor({}, opts);
        }
    }
    else {
        Tensor yc0{ d + xc0 * (c + xc0 * (b + xc0 * a)) },
            yc1{ d + xc1 * (c + xc1 * (b + xc1 * a)) };

        int num_solutions{0};
        if ((yc0 * a).item<double>() >= 0.0) {
            x0 = newtons_method(a, b, c, d, xc0, -1);
            num_solutions++;
        }
        if ((yc0 * yc1).item<double>() <= 0.0) {
            if (torch::abs(yc0).item<double>() < torch::abs(yc1).item<double>()) {
                x1 = newtons_method(a, b, c, d, xc0, 1);
            }
            else
                x1 = newtons_method(a, b, c, d, xc1, -1);
            num_solutions++;
        }
        if ((yc1 * a).item<double>() <= 0.0) {
            x2 = newtons_method(a, b, c, d, xc1, 1);
            num_solutions++;
        }

        switch (num_solutions)
        {
        case 0:
            return torch::tensor({}, opts);
        case 1:
            return x0;
        case 2:
            return torch::cat({ x0, x1 }, 0);
        case 3:
            return torch::cat({ x0, x1, x2 }, 0);
        default:
            return torch::tensor({}, opts);
        }
    }
}

std::vector<Tensor> solve_cubic_backward(Tensor dldz, Tensor ans, Tensor a, Tensor b, Tensor c, Tensor d) {
    Tensor dldd = dldz / (ans * (3 * a * ans + 2 * b) + c);
    Tensor dldc = dldd * ans;
    Tensor dldb = dldc * ans;
    Tensor dlda = dldb * ans;
    return { dlda, dldb, dldc, dldd };
}