#pragma once
#include <array>
#include "Solvers.h"
#include "Constants.h"

struct Yarn {
	double R;
	Tensor rho, Y, B;
	Tensor ks, kb;

	Yarn() = default;
	Yarn(double _R, Tensor _rho, Tensor _Y, Tensor _B):
		R{ _R }, rho{ _rho }, Y{ _Y }, B{ _B }{
		ks = Y * PI * pow(R, 2);
		kb = B * PI * pow(R, 2);
	}
};

struct InterYarn {
	std::array<Yarn*, 2> yarns;
	Tensor mu, kf, df;
	Tensor kc;
	Tensor S, kx;

	InterYarn(std::array<Yarn*, 2> _yarns, Tensor _mu, Tensor _kf, Tensor _df, Tensor _kc, Tensor _S):
		yarns{ _yarns }, mu{ _mu }, kf{ _kf }, df{ _df }, kc{_kc}, S{ _S }{
		kx = S * yarns[0]->R * yarns[1]->R;
	}
};