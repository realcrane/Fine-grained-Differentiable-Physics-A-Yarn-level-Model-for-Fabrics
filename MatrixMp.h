#pragma once
#include "Solvers.h"

Tensor To_Sparse_Mat(int width, int length, const std::vector<Tensor>& M_LL, const std::vector<Tensor>& V_LE, const std::vector<Tensor>& V_EL, const std::vector<Tensor>& S_EE);

Tensor To_Sparse_Mat(int width, int length, const Tensor& M_LL, const Tensor& V_LE, const Tensor& V_EL, const Tensor& S_EE);

Tensor To_Sparse_Vec(const std::vector<Tensor>& V_L, const std::vector<Tensor>& V_E);

Tensor To_Sparse_Vec(const Tensor& V_L, const Tensor& V_E);

