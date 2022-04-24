#include "matrixMp.h"

Tensor To_Sparse_Mat(int width, int length, const std::vector<Tensor>& M_LL, const std::vector<Tensor>& V_LE, const std::vector<Tensor>& V_EL, const std::vector<Tensor>& S_EE)
{
	int num_nodes = width * length;
	int num_cros_nodes = (width - 2) * (length - 2);

	std::vector<Tensor> cols;
	for (int i = 0; i < num_nodes; ++i) {
		std::vector<Tensor> rows;
		for (int j = 0; j < num_nodes; ++j) {
			rows.push_back(M_LL[j + i * num_nodes]);
			if (j == num_nodes - 1) {
				for (int row_i = 0; row_i < (num_cros_nodes * 2); ++row_i) {
					rows.push_back(V_LE[i * num_cros_nodes * 2 + row_i]);
				}
			}
		}
		cols.push_back(torch::cat(rows, 1));
		if (i == num_nodes - 1) {
			std::vector<Tensor> cols_bot;
			for (int col_i = 0; col_i < (num_cros_nodes * 2); ++col_i) {
				std::vector<Tensor> rows_bot;
				for (int col_j = 0; col_j < num_nodes; ++col_j) {
					rows_bot.push_back(V_EL[col_j * num_cros_nodes * 2 + col_i]);
					if (col_j == num_nodes - 1) {
						for (int EE_i = 0; EE_i < (num_cros_nodes * 2); ++EE_i) {
							rows_bot.push_back(S_EE[EE_i * num_cros_nodes * 2 + col_i]);
						}
					}
				}
				cols_bot.push_back(torch::cat(rows_bot, 1));
			}
			cols.push_back(torch::cat(cols_bot, 0));
		}
	}
	return torch::cat(cols, 0);
}

Tensor To_Sparse_Mat(int width, int length, const Tensor& M_LL, const Tensor& V_LE, const Tensor& V_EL, const Tensor& S_EE) {
	int num_nodes = width * length;
	int num_cros_nodes = (width - 2) * (length - 2);
	int dim_matrix = num_nodes * 3 + num_cros_nodes * 2;
	int col_vector = num_cros_nodes * 2;

	Tensor A{ torch::zeros({dim_matrix, dim_matrix}, opts) };
	double* A_ptr = A.data_ptr<double>();

	auto M_LL_acc = M_LL.accessor<double, 3>();
	auto V_LE_acc = V_LE.accessor<double, 3>();
	auto V_EL_acc = V_EL.accessor<double, 3>();
	auto S_EE_acc = S_EE.accessor<double, 3>();
	
	for (int j = 0; j < num_nodes; ++j) {
		for (int i = 0; i < num_nodes; ++i) {
			int index_MM = i + j * num_nodes;
			for (int u = 0; u < 3; ++u)
				for (int v = 0; v < 3; ++v) {
					int index_A_u = i * 3 + u;
					int index_A_v = j * 3 + v;
					int index_A = index_A_u + index_A_v * dim_matrix;
					*(A_ptr + index_A) = M_LL_acc[index_MM][u][v];
				}
		} 
	}

	for (int row_j = 0; row_j < num_nodes; ++row_j)
		for (int col_i = 0; col_i < col_vector; ++col_i) {
			int index_LE = col_i + row_j * col_vector;
			for (int v = 0; v < 3; ++v) {
				int index_A_u = num_nodes * 3 + col_i;
				int index_A_v = row_j * 3 + v;
				int index_A = index_A_u + index_A_v * dim_matrix;
				*(A_ptr + index_A) = V_LE_acc[index_LE][v][0];
			}
		}

	for (int row_j = 0; row_j < col_vector; ++row_j)
		for (int col_i = 0; col_i < num_nodes; ++col_i) {
			int index_EL = row_j + col_i * col_vector;
			for (int u = 0; u < 3; ++u) {
				int index_A_u = col_i * 3 + u;
				int index_A_v = num_nodes * 3 + row_j;
				int index_A = index_A_u + index_A_v * dim_matrix;
				*(A_ptr + index_A) = V_EL_acc[index_EL][0][u];
			}
		}

	for (int scl_j = 0; scl_j < col_vector; ++scl_j)
		for (int scl_i = 0; scl_i < col_vector; ++scl_i) {
			int index_EE = scl_i + scl_j * col_vector;
			int index_A_u = num_nodes * 3 + scl_i;
			int index_A_v = num_nodes * 3 + scl_j;
			int index_A = index_A_u + index_A_v * dim_matrix;
			*(A_ptr + index_A) = S_EE_acc[index_EE][0][0];
		}

	return A;
}

Tensor To_Sparse_Vec(const std::vector<Tensor>& V_L, const std::vector<Tensor>& V_E)
{
	std::vector<Tensor> V_LE;
	V_LE.insert(V_LE.end(), V_L.begin(), V_L.end());
	V_LE.insert(V_LE.end(), V_E.begin(), V_E.end());
	return torch::cat(V_LE, 0);
}

Tensor To_Sparse_Vec(const Tensor& V_L, const Tensor& V_E) {
	return torch::cat({ V_L.view({-1, 1}), V_E.view({-1, 1}) }, 0);
}
