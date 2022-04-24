#include "PhysicsCloth.h"

void PhysicsCloth::InitMV()
{
	omp_set_num_threads(NThrd);
	
	for (int t = 0; t < NThrd; ++t) {
		M_LLs[t] = torch::zeros({ env->cloth->num_LLD, 3, 3 }, opts);
		M_ELs[t] = torch::zeros({ env->cloth->num_ELD, 1, 3 }, opts);
		M_LEs[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);
		M_EEs[t] = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);

		F_Ls[t] = torch::zeros({ env->cloth->num_nodes, 3, 1 }, opts);
		F_Es[t] = torch::zeros({ env->cloth->num_cros_nodes, 2, 1 }, opts);

		F_LL_Poss[t] = torch::zeros({ env->cloth->num_LLD, 3, 3 }, opts);
		F_EL_Poss[t] = torch::zeros({ env->cloth->num_ELD, 1, 3 }, opts);
		F_LE_Poss[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);
		F_EE_Poss[t] = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);

		F_LL_Vels[t] = torch::zeros({ env->cloth->num_LLD, 3, 3 }, opts);
		F_EL_Vels[t] = torch::zeros({ env->cloth->num_ELD, 1, 3 }, opts);
		F_LE_Vels[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);
		F_EE_Vels[t] = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);

		StrhBend_NC_Ls[t] = torch::zeros({ env->cloth->num_nodes, 3, 1 }, opts);
		StrhBend_C_Ls[t] = torch::zeros({ env->cloth->num_nodes, 3, 1 }, opts);

		StrhBend_NC_LEs[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);
		StrhBend_C_LEs[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);

		F_E_is_Slides[t] = torch::zeros({ env->cloth->num_cros_nodes, 2, 1 }, opts);
	}

	F_E_No_Slide = torch::zeros({ env->cloth->num_cros_nodes, 2, 1 }, opts);
	F_EE_Pos_No_Slide = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);
	F_EE_Vel_No_Slide = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);

	Pos_L = std::vector<Tensor>(env->cloth->num_nodes, ZERO31);
	Pos_E = std::vector<Tensor>(env->cloth->num_cros_nodes, ZERO21);

	Vel_L = std::vector<Tensor>(env->cloth->num_nodes, ZERO31);
	Vel_E = std::vector<Tensor>(env->cloth->num_cros_nodes, ZERO21);
}

void PhysicsCloth::FillMV() 
{
	for (int i = 0; i < env->cloth->width; ++i)
		for (int j = 0; j < env->cloth->length; ++j) {
			VecPosVel(i, j);
		}

#pragma omp parallel
	{

#pragma omp for collapse(2) nowait
		for (int i = 0; i < env->cloth->width; ++i)
			for (int j = 0; j < env->cloth->length; ++j) {
				int ID = omp_get_thread_num();
				Constraint_F(i, j, F_Ls[ID]);
				Constraint_D(i, j, F_LL_Poss[ID]);
			}

#pragma omp for nowait
		for (int i = 0; i < env->cloth->edges.size(); ++i) {
			int ID = omp_get_thread_num();
			Mass(env->cloth->edges[i], M_LLs[ID], M_ELs[ID], M_LEs[ID], M_EEs[ID]);
			Gravity_F(env->cloth->edges[i], F_Ls[ID], F_Es[ID]);
			Stretch_F(env->cloth->edges[i], F_Ls[ID], F_Es[ID], StrhBend_NC_Ls[ID], StrhBend_C_Ls[ID]);
			Inertia_F(env->cloth->edges[i], F_Ls[ID], F_Es[ID]);
			MDotVDot_F(env->cloth->edges[i], F_Ls[ID], F_Es[ID]);
			ParaColli_F(env->cloth->edges[i], F_Es[ID]); // Use the same property
			Gravity_D(env->cloth->edges[i], F_LL_Poss[ID], F_EL_Poss[ID], F_LE_Poss[ID], F_EE_Poss[ID]);
			Stretch_D(env->cloth->edges[i], F_LL_Poss[ID], F_EL_Poss[ID], F_LE_Poss[ID], F_EE_Poss[ID], StrhBend_NC_LEs[ID], StrhBend_C_LEs[ID]);
			Inertia_D_Pos(env->cloth->edges[i], F_LL_Poss[ID], F_EL_Poss[ID], F_LE_Poss[ID], F_EE_Poss[ID]);
			Inertia_D_Vel(env->cloth->edges[i], F_LL_Vels[ID], F_EL_Vels[ID], F_LE_Vels[ID], F_EE_Vels[ID]);
			MDotVDot_D_Pos(env->cloth->edges[i], F_LL_Poss[ID], F_EL_Poss[ID], F_LE_Poss[ID], F_EE_Poss[ID]);
			MDotVDot_D_Vel(env->cloth->edges[i], F_LL_Vels[ID], F_EL_Vels[ID], F_LE_Vels[ID], F_EE_Vels[ID]);
			ParaColli_D(env->cloth->edges[i], F_EE_Poss[ID]); // Use the same property
		}

#pragma omp for nowait
		for (int i = 0; i < env->cloth->bend_segs.size(); ++i) {
			int ID = omp_get_thread_num();
			Bending_F(env->cloth->bend_segs[i], F_Ls[ID], F_Es[ID], StrhBend_NC_Ls[ID]);
			Bending_D(env->cloth->bend_segs[i], F_LL_Poss[ID], F_EL_Poss[ID], F_LE_Poss[ID], F_EE_Poss[ID], StrhBend_NC_LEs[ID]);
			Crimp_Bending_F(env->cloth->crimp_bend_segs[i], StrhBend_C_Ls[ID]);
			Crimp_Bending_D(env->cloth->crimp_bend_segs[i], StrhBend_C_LEs[ID]);
		}

#pragma omp for
		for (int i = 0; i < env->cloth->faces.size(); ++i) {
			int ID = omp_get_thread_num();
			Wind_F(env->cloth->faces[i], F_Ls[ID]);
		}

#pragma omp single
		{
			for (int t = 1; t < NThrd; ++t) {
				StrhBend_NC_Ls[0] = StrhBend_NC_Ls[0] + StrhBend_NC_Ls[t];
				StrhBend_C_Ls[0] = StrhBend_C_Ls[0] + StrhBend_C_Ls[t];

				StrhBend_NC_LEs[0] = StrhBend_NC_LEs[0] + StrhBend_NC_LEs[t];
				StrhBend_C_LEs[0] = StrhBend_C_LEs[0] + StrhBend_C_LEs[t];
			}
			for (int t = 0; t < NThrd; ++t) {
				F_E_No_Slide = F_E_No_Slide + F_Es[t];

				F_EE_Pos_No_Slide = F_EE_Pos_No_Slide + F_EE_Poss[t];
				F_EE_Vel_No_Slide = F_EE_Vel_No_Slide + F_EE_Vels[t];
			}
		}

#pragma omp for collapse(2)
		for (int i = 1; i < env->cloth->length - 1; ++i)
			for (int j = 1; j < env->cloth->width - 1; ++j) {
				int ID = omp_get_thread_num();
				std::string node_idx{ std::to_string(j) + "," + std::to_string(i) };
				ConSlideFriction(env->cloth->nodes[node_idx], F_E_is_Slides[ID],
					F_E_No_Slide, F_EE_Pos_No_Slide, F_EE_Vel_No_Slide,
					F_Es[ID], F_EE_Poss[ID], F_EE_Vels[ID],
					StrhBend_C_Ls[0], StrhBend_NC_Ls[0], StrhBend_C_LEs[0], StrhBend_NC_LEs[0]);
			}

#pragma omp for
		for (int i = 0; i < env->cloth->shear_segs.size(); ++i) {
			int ID = omp_get_thread_num();

			Shear_F(env->cloth->shear_segs[i], F_Ls[ID]);
			Shear_D(env->cloth->shear_segs[i], F_LL_Poss[ID]);
		}

	} // End Parallel Region

	for (int t = 1; t < NThrd; ++t) {
		M_LLs[0] = M_LLs[0] + M_LLs[t];
		M_ELs[0] = M_ELs[0] + M_ELs[t];
		M_LEs[0] = M_LEs[0] + M_LEs[t];
		M_EEs[0] = M_EEs[0] + M_EEs[t];

		F_Ls[0] = F_Ls[0] + F_Ls[t];
		F_Es[0] = F_Es[0] + F_Es[t];

		F_LL_Poss[0] = F_LL_Poss[0] + F_LL_Poss[t];
		F_EL_Poss[0] = F_EL_Poss[0] + F_EL_Poss[t];
		F_LE_Poss[0] = F_LE_Poss[0] + F_LE_Poss[t];
		F_EE_Poss[0] = F_EE_Poss[0] + F_EE_Poss[t];

		F_LL_Vels[0] = F_LL_Vels[0] + F_LL_Vels[t];
		F_EL_Vels[0] = F_EL_Vels[0] + F_EL_Vels[t];
		F_LE_Vels[0] = F_LE_Vels[0] + F_LE_Vels[t];
		F_EE_Vels[0] = F_EE_Vels[0] + F_EE_Vels[t];

		F_E_is_Slides[0] = F_E_is_Slides[0] + F_E_is_Slides[t];
	}
}

void PhysicsCloth::ResetMV()
{
	for (int t = 0; t < NThrd; ++t) {
		M_LLs[t] = torch::zeros({ env->cloth->num_LLD, 3, 3 }, opts);
		M_ELs[t] = torch::zeros({ env->cloth->num_ELD, 1, 3 }, opts);
		M_LEs[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);
		M_EEs[t] = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);

		F_Ls[t] = torch::zeros({ env->cloth->num_nodes, 3, 1 }, opts);
		F_Es[t] = torch::zeros({ env->cloth->num_cros_nodes, 2, 1 }, opts);

		F_LL_Poss[t] = torch::zeros({ env->cloth->num_LLD, 3, 3 }, opts);
		F_EL_Poss[t] = torch::zeros({ env->cloth->num_ELD, 1, 3 }, opts);
		F_LE_Poss[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);
		F_EE_Poss[t] = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);

		F_LL_Vels[t] = torch::zeros({ env->cloth->num_LLD, 3, 3 }, opts);
		F_EL_Vels[t] = torch::zeros({ env->cloth->num_ELD, 1, 3 }, opts);
		F_LE_Vels[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);
		F_EE_Vels[t] = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);

		StrhBend_NC_Ls[t] = torch::zeros({ env->cloth->num_nodes, 3, 1 }, opts);
		StrhBend_C_Ls[t] = torch::zeros({ env->cloth->num_nodes, 3, 1 }, opts);

		StrhBend_NC_LEs[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);
		StrhBend_C_LEs[t] = torch::zeros({ env->cloth->num_ELD, 3, 1 }, opts);

		F_E_is_Slides[t] = torch::zeros({ env->cloth->num_cros_nodes, 2, 1 }, opts);
	}

	F_E_No_Slide = torch::zeros({ env->cloth->num_cros_nodes, 2, 1 }, opts);
	F_EE_Pos_No_Slide = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);
	F_EE_Vel_No_Slide = torch::zeros({ env->cloth->num_EED, 1, 1 }, opts);

	std::fill(Pos_L.begin(), Pos_L.end(), ZERO31);
	std::fill(Pos_E.begin(), Pos_E.end(), ZERO21);

	std::fill(Vel_L.begin(), Vel_L.end(), ZERO31);
	std::fill(Vel_E.begin(), Vel_E.end(), ZERO21);
}

int PhysicsCloth::find_index(int F_u, int F_v, int D_u, int D_v, std::string D_type, std::string uv)
{
	if (D_type == "LL")
		return (F_u + F_v * env->cloth->width) * env->cloth->num_nodes + (D_u + D_v * env->cloth->width);
	if (D_type == "EL")
		return (F_u + F_v * env->cloth->width) * env->cloth->num_cros_nodes * 2 + (D_u - 1 + (D_v - 1) * (env->cloth->width - 2)) * 2;
	if (D_type == "EE") {
		if (uv == "uu")
			return (F_u - 1 + (F_v - 1) * (env->cloth->width - 2)) * 2 * env->cloth->num_cros_nodes * 2 + (D_u - 1 + (D_v - 1) * (env->cloth->width - 2)) * 2;
		if (uv == "vv")
			return ((F_u - 1 + (F_v - 1) * (env->cloth->width - 2)) * 2 + 1) * env->cloth->num_cros_nodes * 2 + (D_u - 1 + (D_v - 1) * (env->cloth->width - 2)) * 2 + 1;
		if (uv == "uv")
			return (F_u - 1 + (F_v - 1) * (env->cloth->width - 2)) * env->cloth->num_cros_nodes * 2 + (D_u - 1 + (D_v - 1) * (env->cloth->width - 2)) * 2 + 1;
		if (uv == "vu")
			return (F_u - 1 + (F_v - 1) * (env->cloth->width - 2) + 1) * env->cloth->num_cros_nodes * 2 + (D_u - 1 + (D_v - 1) * (env->cloth->width - 2)) * 2;
	}
	return 0;
}

void PhysicsCloth::VecPosVel(int u, int v)
{
	int L_idx{ u + v * env->cloth->width };
	std::string idx{ std::to_string(u) + "," + std::to_string(v) };

	Pos_L.at(L_idx) = env->cloth->nodes[idx].LPos;
	Vel_L.at(L_idx) = env->cloth->nodes[idx].LVel;
	if (!(env->cloth->nodes[idx].is_edge)) {
		int E_idx = u - 1 + (v - 1) * (env->cloth->width - 2);
		Pos_E.at(E_idx) = env->cloth->nodes[idx].EPos;
		Vel_E.at(E_idx) = env->cloth->nodes[idx].EVel;
	}
}

void PhysicsCloth::Constraint_F(int u, int v, Tensor& F_L)
{
	std::string n_idx = std::to_string(u) + "," + std::to_string(v);
	int F_idx{ u + v * env->cloth->width };

	if (env->cloth->nodes[n_idx].is_fixed)
		F_L[F_idx] = F_L[F_idx] - env->cloth->handle_stiffness * (env->cloth->nodes[n_idx].LPos - env->cloth->nodes[n_idx].LPosFix);
}

void PhysicsCloth::Constraint_D(int u, int v, Tensor& F_LL_Pos)
{
	std::string idx = std::to_string(u) + "," + std::to_string(v);
	int F_LL_idx{ find_index(u, v, u, v, "LL", "NONE") };

	if (env->cloth->nodes[idx].is_fixed)
		F_LL_Pos[F_LL_idx] = F_LL_Pos[F_LL_idx] - env->cloth->handle_stiffness * EYE3;
}

void PhysicsCloth::Mass(const Edge& e, Tensor& M_LL, Tensor& M_EL, Tensor& M_LE, Tensor& M_EE)
{
	int L0L0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L1L0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L0L1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") },
		L1L1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") };

	Tensor coeff{ (1.0 / 6.0) * e.yarn.rho * e.delta_uv };
	Tensor L0L0_mass{ 2.0 * coeff * EYE3 }, L1L1_mass{ L0L0_mass },
		L0L1_mass{ coeff * EYE3 }, L1L0_mass{ L0L1_mass };
	Tensor L0E0_mass{ -2.0 * coeff * e.w }, L1E1_mass{ L0E0_mass };
	Tensor E0L0_mass{ -2.0 * coeff * torch::transpose(e.w, 0, 1) }, E1L1_mass{ E0L0_mass };
	Tensor E0E0_mass{ 2.0 * coeff * torch::mm(torch::transpose(e.w, 0, 1), e.w) }, E1E1_mass{ E0E0_mass };
	Tensor L0E1_mass{ -1.0 * coeff * e.w }, L1E0_mass{ L0E1_mass };
	Tensor E1L0_mass{ -1.0 * coeff * torch::transpose(e.w, 0, 1) }, E0L1_mass{ E1L0_mass };
	Tensor E0E1_mass{ 0.5 * E0E0_mass }, E1E0_mass = { E0E1_mass };

	M_LL[L0L0_idx] = M_LL[L0L0_idx] + L0L0_mass;
	M_LL[L1L0_idx] = M_LL[L1L0_idx] + L1L0_mass;
	M_LL[L0L1_idx] = M_LL[L0L1_idx] + L0L1_mass;
	M_LL[L1L1_idx] = M_LL[L1L1_idx] + L1L1_mass;

	if (!(e.n0->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			M_LE[L0E0_idx] = M_LE[L0E0_idx] + L0E0_mass;
			M_EL[L0E0_idx] = M_EL[L0E0_idx] + E0L0_mass;
			M_EE[E0E0_idx] = M_EE[E0E0_idx] + E0E0_mass;
		}
		else {
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };

			M_LE[L0E0_idx] = M_LE[L0E0_idx] + L0E0_mass;
			M_EL[L0E0_idx] = M_EL[L0E0_idx] + E0L0_mass;
			M_EE[E0E0_idx] = M_EE[E0E0_idx] + E0E0_mass;
		}
	}

	if (!(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };

			M_LE[L1E1_idx] = M_LE[L1E1_idx] + L1E1_mass;
			M_EL[L1E1_idx] = M_EL[L1E1_idx] + E1L1_mass;
			M_EE[E1E1_idx] = M_EE[E1E1_idx] + E1E1_mass;
		}
		else {
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1 };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };

			M_LE[L1E1_idx] = M_LE[L1E1_idx] + L1E1_mass;
			M_EL[L1E1_idx] = M_EL[L1E1_idx] + E1L1_mass;
			M_EE[E1E1_idx] = M_EE[E1E1_idx] + E1E1_mass;
		}
	}

	if (!(e.n0->is_edge) && !(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			M_LE[L1E0_idx] = M_LE[L1E0_idx] + L1E0_mass;
			M_LE[L0E1_idx] = M_LE[L0E1_idx] + L0E1_mass;
			M_EL[L1E0_idx] = M_EL[L1E0_idx] + E0L1_mass;
			M_EL[L0E1_idx] = M_EL[L0E1_idx] + E1L0_mass;
			M_EE[E0E1_idx] = M_EE[E0E1_idx] + E0E1_mass;
			M_EE[E1E0_idx] = M_EE[E1E0_idx] + E1E0_mass;
		}
		else {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1 };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };

			M_LE[L1E0_idx] = M_LE[L1E0_idx] + L1E0_mass;
			M_LE[L0E1_idx] = M_LE[L0E1_idx] + L0E1_mass;
			M_EL[L1E0_idx] = M_EL[L1E0_idx] + E0L1_mass;
			M_EL[L0E1_idx] = M_EL[L0E1_idx] + E1L0_mass;
			M_EE[E0E1_idx] = M_EE[E0E1_idx] + E0E1_mass;
			M_EE[E1E0_idx] = M_EE[E1E0_idx] + E1E0_mass;
		}
	}
}

void PhysicsCloth::Gravity_F(const Edge& e, Tensor& F_L, Tensor& F_E)
{
	int L_idx_n0{ e.n0->idx_u + e.n0->idx_v * env->cloth->width };
	int L_idx_n1{ e.n1->idx_u + e.n1->idx_v * env->cloth->width };

	F_L[L_idx_n0] = F_L[L_idx_n0] - 0.5 * e.yarn.rho * env->G * e.delta_uv;
	F_L[L_idx_n1] = F_L[L_idx_n1] - 0.5 * e.yarn.rho * env->G * e.delta_uv;

	if (!(e.n0->is_edge)) {
		int E_idx_n0{ (e.n0->idx_u - 1) + (e.n0->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp")
			F_E[E_idx_n0] = F_E[E_idx_n0] - (-0.5) * e.yarn.rho * torch::mm(env->G.transpose(1, 0), e.n0->LPos + e.n1->LPos) * U_OneHot;
		else
			F_E[E_idx_n0] = F_E[E_idx_n0] - (-0.5) * e.yarn.rho * torch::mm(env->G.transpose(1, 0), e.n0->LPos + e.n1->LPos) * V_OneHot;
	}
	if (!e.n1->is_edge) {
		int E_idx_n1{ (e.n1->idx_u - 1) + (e.n1->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp")
			F_E[E_idx_n1] = F_E[E_idx_n1] - 0.5 * e.yarn.rho * torch::mm(env->G.transpose(1, 0), e.n0->LPos + e.n1->LPos) * U_OneHot;
		else
			F_E[E_idx_n1] = F_E[E_idx_n1] - 0.5 * e.yarn.rho * torch::mm(env->G.transpose(1, 0), e.n0->LPos + e.n1->LPos) * V_OneHot;
	}
}

void PhysicsCloth::Gravity_D(const Edge& e, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos)
{
	Tensor L0E0_gravity = 0.5 * e.yarn.rho * env->G, L1E0_gravity = L0E0_gravity;
	Tensor L0E1_gravity = -L0E0_gravity, L1E1_gravity = -L0E0_gravity;
	Tensor E0L0_gravity = L0E0_gravity.transpose(1, 0), E0L1_gravity = E0L0_gravity;
	Tensor E1L0_gravity = L0E1_gravity.transpose(1, 0), E1L1_gravity = E1L0_gravity;

	if (!(e.n0->is_edge)) {
		if (e.edge_type == "warp")
		{
			int L0E0_idx = find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE");

			F_EL_Pos[L0E0_idx] = F_EL_Pos[L0E0_idx] + E0L0_gravity;
			F_LE_Pos[L0E0_idx] = F_LE_Pos[L0E0_idx] + L0E0_gravity;
		}
		else
		{
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };

			F_EL_Pos[L0E0_idx] = F_EL_Pos[L0E0_idx] + E0L0_gravity;
			F_LE_Pos[L0E0_idx] = F_LE_Pos[L0E0_idx] + L0E0_gravity;
		}
	}

	if (!(e.n1->is_edge)) {
		if (e.edge_type == "warp")
		{
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };

			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_gravity;
			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_gravity;
		}
		else
		{
			int L1E1_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1;

			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_gravity;
			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_gravity;
		}
	}

	if (!(e.n0->is_edge) && !(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };

			F_LE_Pos[L1E0_idx] = F_LE_Pos[L1E0_idx] + L1E0_gravity;
			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_gravity;
			F_EL_Pos[L1E0_idx] = F_EL_Pos[L1E0_idx] + E0L1_gravity;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_gravity;
		}
		else {
			int L0E1_idx = find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1;
			int L1E0_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1;

			F_LE_Pos[L1E0_idx] = F_LE_Pos[L1E0_idx] + L1E0_gravity;
			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_gravity;
			F_EL_Pos[L1E0_idx] = F_EL_Pos[L1E0_idx] + E0L1_gravity;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_gravity;
		}
	}
}

void PhysicsCloth::Stretch_F(const Edge& e, Tensor& F_L, Tensor& F_E, Tensor& StrhBend_NC_L, Tensor& StrhBend_C_L)
{
	int L_idx_n0{ e.n0->idx_u + e.n0->idx_v * env->cloth->width };
	int L_idx_n1{ e.n1->idx_u + e.n1->idx_v * env->cloth->width };

	F_L[L_idx_n0] = F_L[L_idx_n0] + e.yarn.ks * (torch::norm(e.w) - 1) * e.d;
	F_L[L_idx_n1] = F_L[L_idx_n1] - e.yarn.ks * (torch::norm(e.w) - 1) * e.d;

	if (e.edge_type == "warp") {
		StrhBend_NC_L[L_idx_n0] = StrhBend_NC_L[L_idx_n0] + e.yarn.ks * (torch::norm(e.w) - 1) * e.d;
		StrhBend_NC_L[L_idx_n1] = StrhBend_NC_L[L_idx_n1] - e.yarn.ks * (torch::norm(e.w) - 1) * e.d;

		StrhBend_C_L[L_idx_n0] = StrhBend_C_L[L_idx_n0] + e.yarn.ks * (torch::norm(e.w) - 1) * e.d;
		StrhBend_C_L[L_idx_n1] = StrhBend_C_L[L_idx_n1] - e.yarn.ks * (torch::norm(e.w) - 1) * e.d;
	}
	else {
		StrhBend_NC_L[L_idx_n0] = StrhBend_NC_L[L_idx_n0] - e.yarn.ks * (torch::norm(e.w) - 1) * e.d;
		StrhBend_NC_L[L_idx_n1] = StrhBend_NC_L[L_idx_n1] + e.yarn.ks * (torch::norm(e.w) - 1) * e.d;

		StrhBend_C_L[L_idx_n0] = StrhBend_C_L[L_idx_n0] - e.yarn.ks * (torch::norm(e.w) - 1) * e.d;
		StrhBend_C_L[L_idx_n1] = StrhBend_C_L[L_idx_n1] + e.yarn.ks * (torch::norm(e.w) - 1) * e.d;
	}

	if (!(e.n0->is_edge)) {
		int E_idx_n0{ (e.n0->idx_u - 1) + (e.n0->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp")
			F_E[E_idx_n0] = F_E[E_idx_n0] - 0.5 * e.yarn.ks * (pow(torch::norm(e.w), 2) - 1) * U_OneHot;
		else
			F_E[E_idx_n0] = F_E[E_idx_n0] - 0.5 * e.yarn.ks * (pow(torch::norm(e.w), 2) - 1) * V_OneHot;
	}
	if (!e.n1->is_edge) {
		int E_idx_n1{ (e.n1->idx_u - 1) + (e.n1->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp")
			F_E[E_idx_n1] = F_E[E_idx_n1] + 0.5 * e.yarn.ks * (pow(torch::norm(e.w), 2) - 1) * U_OneHot;
		else
			F_E[E_idx_n1] = F_E[E_idx_n1] + 0.5 * e.yarn.ks * (pow(torch::norm(e.w), 2) - 1) * V_OneHot;
	}
}

void PhysicsCloth::Stretch_D(const Edge& e, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos, Tensor& StrhBend_NC_LE, Tensor& StrhBend_C_LE)
{
	int L0L0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L1L0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L0L1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") },
		L1L1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") };

	Tensor L1L1_stretch = (e.yarn.ks / e.l) * e.P - (e.yarn.ks / e.delta_uv) * EYE3, L0L0_stretch = L1L1_stretch;
	Tensor L1L0_stretch = -L1L1_stretch, L0L1_stretch = -L1L1_stretch;
	Tensor E1E1_stretch = -e.yarn.ks * (torch::pow(torch::norm(e.w), 2) / e.delta_uv), E0E0_stretch = E1E1_stretch;
	Tensor E1E0_stretch = -E1E1_stretch, E0E1_stretch = -E1E1_stretch;
	Tensor L1E1_stretch = e.yarn.ks * (torch::norm(e.w) / e.delta_uv) * e.d, L0E0_stretch = L1E1_stretch;
	Tensor L1E0_stretch = -L1E1_stretch, L0E1_stretch = -L1E1_stretch;
	Tensor E1L1_stretch = e.yarn.ks / e.delta_uv * e.w.transpose(1, 0), E0L0_stretch = E1L1_stretch;
	Tensor E1L0_stretch = -E1L1_stretch, E0L1_stretch = -E1L1_stretch;

	F_LL_Pos[L0L0_idx] = F_LL_Pos[L0L0_idx] + L0L0_stretch;
	F_LL_Pos[L1L0_idx] = F_LL_Pos[L1L0_idx] + L1L0_stretch;
	F_LL_Pos[L0L1_idx] = F_LL_Pos[L0L1_idx] + L0L1_stretch;
	F_LL_Pos[L1L1_idx] = F_LL_Pos[L1L1_idx] + L1L1_stretch;

	if (!(e.n0->is_edge)) {
		if (e.edge_type == "warp")
		{
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_LE_Pos[L0E0_idx] = F_LE_Pos[L0E0_idx] + L0E0_stretch;
			F_EL_Pos[L0E0_idx] = F_EL_Pos[L0E0_idx] + E0L0_stretch;
			F_EE_Pos[E0E0_idx] = F_EE_Pos[E0E0_idx] + E0E0_stretch;

			StrhBend_NC_LE[L0E0_idx] = StrhBend_NC_LE[L0E0_idx] + L0E0_stretch;
			StrhBend_C_LE[L0E0_idx] = StrhBend_C_LE[L0E0_idx] + L0E0_stretch;
		}
		else
		{
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };

			F_LE_Pos[L0E0_idx] = F_LE_Pos[L0E0_idx] + L0E0_stretch;
			F_EL_Pos[L0E0_idx] = F_EL_Pos[L0E0_idx] + E0L0_stretch;
			F_EE_Pos[E0E0_idx] = F_EE_Pos[E0E0_idx] + E0E0_stretch;

			StrhBend_NC_LE[L0E0_idx] = StrhBend_NC_LE[L0E0_idx] - L0E0_stretch;
			StrhBend_C_LE[L0E0_idx] = StrhBend_C_LE[L0E0_idx] - L0E0_stretch;
		}
	}

	if (!(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };

			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_stretch;
			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_stretch;
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_stretch;

			StrhBend_NC_LE[L1E1_idx] = StrhBend_NC_LE[L1E1_idx] + L1E1_stretch;
			StrhBend_C_LE[L1E1_idx] = StrhBend_C_LE[L1E1_idx] + L1E1_stretch;
		}
		else {
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1 };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };

			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_stretch;
			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_stretch;
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_stretch;

			StrhBend_NC_LE[L1E1_idx] = StrhBend_NC_LE[L1E1_idx] - L1E1_stretch;
			StrhBend_C_LE[L1E1_idx] = StrhBend_C_LE[L1E1_idx] - L1E1_stretch;
		}
	}

	if (!(e.n0->is_edge) && !(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_LE_Pos[L1E0_idx] = F_LE_Pos[L1E0_idx] + L1E0_stretch;
			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_stretch;
			F_EL_Pos[L1E0_idx] = F_EL_Pos[L1E0_idx] + E0L1_stretch;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_stretch;
			F_EE_Pos[E0E1_idx] = F_EE_Pos[E0E1_idx] + E0E1_stretch;
			F_EE_Pos[E1E0_idx] = F_EE_Pos[E1E0_idx] + E1E0_stretch;

			StrhBend_NC_LE[L1E0_idx] = StrhBend_NC_LE[L1E0_idx] + L1E0_stretch;
			StrhBend_NC_LE[L0E1_idx] = StrhBend_NC_LE[L0E1_idx] + L0E1_stretch;
			StrhBend_C_LE[L1E0_idx] = StrhBend_C_LE[L1E0_idx] + L1E0_stretch;
			StrhBend_C_LE[L0E1_idx] = StrhBend_C_LE[L0E1_idx] + L0E1_stretch;
		}
		else {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1 };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };

			F_LE_Pos[L1E0_idx] = F_LE_Pos[L1E0_idx] + L1E0_stretch;
			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_stretch;
			F_EL_Pos[L1E0_idx] = F_EL_Pos[L1E0_idx] + E0L1_stretch;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_stretch;
			F_EE_Pos[E0E1_idx] = F_EE_Pos[E0E1_idx] + E0E1_stretch;
			F_EE_Pos[E1E0_idx] = F_EE_Pos[E1E0_idx] + E1E0_stretch;

			StrhBend_NC_LE[L1E0_idx] = StrhBend_NC_LE[L1E0_idx] - L1E0_stretch;
			StrhBend_NC_LE[L0E1_idx] = StrhBend_NC_LE[L0E1_idx] - L0E1_stretch;
			StrhBend_C_LE[L1E0_idx] = StrhBend_C_LE[L1E0_idx] - L1E0_stretch;
			StrhBend_C_LE[L0E1_idx] = StrhBend_C_LE[L0E1_idx] - L0E1_stretch;
		}
	}
}

void PhysicsCloth::Bending_F(const Bend_Seg& b, Tensor& F_L, Tensor& F_E, Tensor& StrhBend_NC_L)
{
	int L2_idx{ b.n2->idx_u + b.n2->idx_v * env->cloth->width },
		L0_idx{ b.n0->idx_u + b.n0->idx_v * env->cloth->width },
		L1_idx{ b.n1->idx_u + b.n1->idx_v * env->cloth->width };

	Tensor F_x1 = ZERO31, F_x2 = ZERO31;

	if ((torch::sin(b.theta)).item<double>() != 0) {
		F_x1 = ((-2 * b.yarn.kb * b.theta) / (b.l1 * b.delta_uv * torch::sin(b.theta))) * torch::mm(b.P1, b.d2);
		F_x2 = ((-2 * b.yarn.kb * b.theta) / (b.l2 * b.delta_uv * torch::sin(b.theta))) * torch::mm(b.P2, b.d1);
	}

	Tensor F_u2 = ((-b.yarn.kb * b.theta) / pow(b.delta_uv, 2)).view({ 1 });
	Tensor F_u1 = -1 * F_u2, F_u0 = ZERO1;
	Tensor F_x0 = -(F_x1 + F_x2);

	F_L[L2_idx] = F_L[L2_idx] + F_x2;
	F_L[L0_idx] = F_L[L0_idx] + F_x0;
	F_L[L1_idx] = F_L[L1_idx] + F_x1;

	if (b.seg_type == "warp") {
		StrhBend_NC_L[L2_idx] = StrhBend_NC_L[L2_idx] + F_x2;
		StrhBend_NC_L[L0_idx] = StrhBend_NC_L[L0_idx] + F_x0;
		StrhBend_NC_L[L1_idx] = StrhBend_NC_L[L1_idx] + F_x1;
	}
	else
	{
		StrhBend_NC_L[L2_idx] = StrhBend_NC_L[L2_idx] - F_x2;
		StrhBend_NC_L[L0_idx] = StrhBend_NC_L[L0_idx] - F_x0;
		StrhBend_NC_L[L1_idx] = StrhBend_NC_L[L1_idx] - F_x1;
	}

	if (!(b.n2->is_edge)) {
		int E2_idx = ((b.n2->idx_u - 1) + (b.n2->idx_v - 1) * (env->cloth->width - 2));
		if (b.seg_type == "warp") {
			F_E[E2_idx] = F_E[E2_idx] + F_u2 * U_OneHot;
		}
		else {
			F_E[E2_idx] = F_E[E2_idx] + F_u2 * V_OneHot;
		}
	}

	if (!(b.n0->is_edge)) {
		int E0_idx = ((b.n0->idx_u - 1) + (b.n0->idx_v - 1) * (env->cloth->width - 2));
		if (b.seg_type == "warp") {
			F_E[E0_idx] = F_E[E0_idx] + F_u0 * U_OneHot;
		}
		else {
			F_E[E0_idx] = F_E[E0_idx] + F_u0 * V_OneHot;
		}
	}

	if (!(b.n1->is_edge)) {
		int E1_idx = ((b.n1->idx_u - 1) + (b.n1->idx_v - 1) * (env->cloth->width - 2));
		if (b.seg_type == "warp") {
			F_E[E1_idx] = F_E[E1_idx] + F_u1 * U_OneHot;
		}
		else {
			F_E[E1_idx] = F_E[E1_idx] + F_u1 * V_OneHot;
		}
	}
}

void PhysicsCloth::Bending_D(const Bend_Seg& b, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos, Tensor& StrhBend_NC_LE)
{
	int L2L2_idx{ find_index(b.n2->idx_u, b.n2->idx_v, b.n2->idx_u, b.n2->idx_v, "LL", "NONE") },
		L2L0_idx{ find_index(b.n2->idx_u, b.n2->idx_v, b.n0->idx_u, b.n0->idx_v, "LL", "NONE") },
		L2L1_idx{ find_index(b.n2->idx_u, b.n2->idx_v, b.n1->idx_u, b.n1->idx_v, "LL", "NONE") },
		L0L2_idx{ find_index(b.n0->idx_u, b.n0->idx_v, b.n2->idx_u, b.n2->idx_v, "LL", "NONE") },
		L0L0_idx{ find_index(b.n0->idx_u, b.n0->idx_v, b.n0->idx_u, b.n0->idx_v, "LL", "NONE") },
		L0L1_idx{ find_index(b.n0->idx_u, b.n0->idx_v, b.n1->idx_u, b.n1->idx_v, "LL", "NONE") },
		L1L2_idx{ find_index(b.n1->idx_u, b.n1->idx_v, b.n2->idx_u, b.n2->idx_v, "LL", "NONE") },
		L1L0_idx{ find_index(b.n1->idx_u, b.n1->idx_v, b.n0->idx_u, b.n0->idx_v, "LL", "NONE") },
		L1L1_idx{ find_index(b.n1->idx_u, b.n1->idx_v, b.n1->idx_u, b.n1->idx_v, "LL", "NONE") };

	Tensor L1L1_bend = ZERO33, L1L2_bend = L1L1_bend,
		L2L1_bend = L1L1_bend, L2L2_bend = L1L1_bend;
	Tensor L1E1_bend = ZERO31, L2E1_bend = ZERO31;
	Tensor E1L1_bend = ZERO13, E1L2_bend = ZERO13;

	if ((torch::sin(b.theta)).item<double>() != 0) {
		L1L1_bend = (2 * b.yarn.kb) / (pow(b.l1, 2) * b.delta_uv * torch::sin(b.theta)) *
			(b.theta * (torch::mm(torch::mm(b.P1, b.d2), b.d1.transpose(1, 0)) +
				(torch::cos(b.theta) / pow(torch::sin(b.theta), 2)) * torch::mm(torch::mm(torch::mm(b.P1, b.d2), b.d2.transpose(1, 0)), b.P1) +
				torch::cos(b.theta) * b.P1 + torch::mm(torch::mm(b.d1, b.d2.transpose(1, 0)), b.P1)) -
				(1 / torch::sin(b.theta)) * torch::mm(torch::mm(torch::mm(b.P1, b.d2), b.d2.transpose(1, 0)), b.P1));

		L1L2_bend = (-2 * b.yarn.kb) / (b.l1 * b.l2 * b.delta_uv * torch::sin(b.theta)) *
			torch::mm(b.theta * (b.P1 - (torch::cos(b.theta)) / pow(torch::sin(b.theta), 2) * torch::mm(torch::mm(b.P1, b.d2), b.d1.transpose(1, 0)))
				+ (1 / torch::sin(b.theta)) * torch::mm(torch::mm(b.P1, b.d2), b.d1.transpose(1, 0)), b.P2);

		L2L1_bend = (-2 * b.yarn.kb) / (b.l1 * b.l2 * b.delta_uv * torch::sin(b.theta)) *
			torch::mm(b.theta * (b.P2 - (torch::cos(b.theta) / pow(torch::sin(b.theta), 2)) * torch::mm(torch::mm(b.P2, b.d1), b.d2.transpose(1, 0))) +
				(1 / torch::sin(b.theta)) * torch::mm(torch::mm(b.P2, b.d1), b.d2.transpose(1, 0)), b.P1);

		L2L2_bend = (2 * b.yarn.kb) / (pow(b.l2, 2) * b.delta_uv * torch::sin(b.theta)) *
			(b.theta * (torch::mm(torch::mm(b.P2, b.d1), b.d2.transpose(1, 0)) +
				(torch::cos(b.theta) / pow(torch::sin(b.theta), 2)) * torch::mm(torch::mm(torch::mm(b.P2, b.d1), b.d1.transpose(1, 0)), b.P2) +
				torch::cos(b.theta) * b.P2 + torch::mm(torch::mm(b.d2, b.d1.transpose(1, 0)), b.P2)) -
				(1 / torch::sin(b.theta)) * torch::mm(torch::mm(torch::mm(b.P2, b.d1), b.d1.transpose(1, 0)), b.P2));

		L1E1_bend = ((2 * b.yarn.kb * b.theta) / (b.l1 * pow(b.delta_uv, 2) * torch::sin(b.theta))) * torch::mm(b.P1, b.d2);
		L2E1_bend = ((2 * b.yarn.kb * b.theta) / (b.l2 * pow(b.delta_uv, 2) * torch::sin(b.theta))) * torch::mm(b.P2, b.d1);
		E1L1_bend = ((2 * b.yarn.kb * b.theta) / (b.l1 * pow(b.delta_uv, 2) * torch::sin(b.theta))) * torch::mm(b.d2.transpose(1, 0), b.P1);
		E1L2_bend = ((2 * b.yarn.kb * b.theta) / (b.l2 * pow(b.delta_uv, 2) * torch::sin(b.theta))) * torch::mm(b.d1.transpose(1, 0), b.P2);
	}

	Tensor L1L0_bend = -(L1L1_bend + L1L2_bend);
	Tensor L2L0_bend = -(L2L1_bend + L2L2_bend);
	Tensor L0L1_bend = -(L1L1_bend + L2L1_bend);
	Tensor L0L2_bend = -(L1L2_bend + L2L2_bend);
	Tensor L0L0_bend = -(L1L0_bend + L2L0_bend);

	Tensor E1E1_bend = (-2 * b.yarn.kb * pow(b.theta, 2)) / pow(b.delta_uv, 3);
	Tensor E2E2_bend = E1E1_bend;
	Tensor E1E2_bend = -E1E1_bend, E2E1_bend = -E1E1_bend;

	Tensor L1E2_bend = -L1E1_bend;
	Tensor L2E2_bend = -L2E1_bend;

	Tensor L0E1_bend = -(L1E1_bend + L2E1_bend);
	Tensor L0E2_bend = -L0E1_bend;

	Tensor E2L1_bend = -E1L1_bend;
	Tensor E2L2_bend = -E1L2_bend;

	Tensor E1L0_bend = -(E1L1_bend + E1L2_bend);
	Tensor E2L0_bend = -E1L0_bend;

	F_LL_Pos[L2L2_idx] = F_LL_Pos[L2L2_idx] + L2L2_bend;
	F_LL_Pos[L2L0_idx] = F_LL_Pos[L2L0_idx] + L2L0_bend;
	F_LL_Pos[L2L1_idx] = F_LL_Pos[L2L1_idx] + L2L1_bend;

	F_LL_Pos[L0L2_idx] = F_LL_Pos[L0L2_idx] + L0L2_bend;
	F_LL_Pos[L0L0_idx] = F_LL_Pos[L0L0_idx] + L0L0_bend;
	F_LL_Pos[L0L1_idx] = F_LL_Pos[L0L1_idx] + L0L1_bend;

	F_LL_Pos[L1L2_idx] = F_LL_Pos[L1L2_idx] + L1L2_bend;
	F_LL_Pos[L1L0_idx] = F_LL_Pos[L1L0_idx] + L1L0_bend;
	F_LL_Pos[L1L1_idx] = F_LL_Pos[L1L1_idx] + L1L1_bend;

	if (!(b.n2->is_edge)) {
		if (b.seg_type == "warp")
		{
			int L2E2_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE");
			int E2E2_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n2->idx_u, b.n2->idx_v, "EE", "uu");

			F_LE_Pos[L2E2_idx] = F_LE_Pos[L2E2_idx] + L2E2_bend;
			F_EL_Pos[L2E2_idx] = F_EL_Pos[L2E2_idx] + E2L2_bend;
			F_EE_Pos[E2E2_idx] = F_EE_Pos[E2E2_idx] + E2E2_bend;

			StrhBend_NC_LE[L2E2_idx] = StrhBend_NC_LE[L2E2_idx] + L2E2_bend;
		}
		else
		{
			int L2E2_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE") + 1;
			int E2E2_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n2->idx_u, b.n2->idx_v, "EE", "vv");

			F_LE_Pos[L2E2_idx] = F_LE_Pos[L2E2_idx] + L2E2_bend;
			F_EL_Pos[L2E2_idx] = F_EL_Pos[L2E2_idx] + E2L2_bend;
			F_EE_Pos[E2E2_idx] = F_EE_Pos[E2E2_idx] + E2E2_bend;

			StrhBend_NC_LE[L2E2_idx] = StrhBend_NC_LE[L2E2_idx] - L2E2_bend;
		}
	}

	if (!(b.n1->is_edge)) {
		if (b.seg_type == "warp")
		{
			int L1E1_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE");
			int E1E1_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n1->idx_u, b.n1->idx_v, "EE", "uu");

			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_bend;
			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_bend;
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_bend;

			StrhBend_NC_LE[L1E1_idx] = StrhBend_NC_LE[L1E1_idx] + L1E1_bend;
		}
		else
		{
			int L1E1_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE") + 1;
			int E1E1_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n1->idx_u, b.n1->idx_v, "EE", "vv");

			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_bend;
			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_bend;
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_bend;

			StrhBend_NC_LE[L1E1_idx] = StrhBend_NC_LE[L1E1_idx] - L1E1_bend;
		}
	}

	if (!(b.n2->is_edge) && !(b.n0->is_edge)) {
		if (b.seg_type == "warp")
		{
			int L2E0_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n0->idx_u, b.n0->idx_v, "EL", "NONE");
			int L0E2_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE");
			int E2E0_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n0->idx_u, b.n0->idx_v, "EE", "uu");
			int E0E2_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n2->idx_u, b.n2->idx_v, "EE", "uu");

			F_LE_Pos[L0E2_idx] = F_LE_Pos[L0E2_idx] + L0E2_bend;
			F_EL_Pos[L2E0_idx] = F_EL_Pos[L2E0_idx] + E2L0_bend;

			StrhBend_NC_LE[L0E2_idx] = StrhBend_NC_LE[L0E2_idx] + L0E2_bend;
		}
		else {
			int L2E0_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n0->idx_u, b.n0->idx_v, "EL", "NONE") + 1;
			int L0E2_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE") + 1;
			int E2E0_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n0->idx_u, b.n0->idx_v, "EE", "vv");
			int E0E2_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n2->idx_u, b.n2->idx_v, "EE", "vv");

			F_LE_Pos[L0E2_idx] = F_LE_Pos[L0E2_idx] + L0E2_bend;
			F_EL_Pos[L2E0_idx] = F_EL_Pos[L2E0_idx] + E2L0_bend;

			StrhBend_NC_LE[L0E2_idx] = StrhBend_NC_LE[L0E2_idx] - L0E2_bend;
		}
	}

	if (!(b.n0->is_edge) && !(b.n1->is_edge)) {
		if (b.seg_type == "warp")
		{
			int L0E1_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE");
			int L1E0_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n0->idx_u, b.n0->idx_v, "EL", "NONE");
			int E0E1_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n1->idx_u, b.n1->idx_v, "EE", "uu");
			int E1E0_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n0->idx_u, b.n0->idx_v, "EE", "uu");

			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_bend;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_bend;

			StrhBend_NC_LE[L0E1_idx] = StrhBend_NC_LE[L0E1_idx] + L0E1_bend;
		}
		else {
			int L0E1_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE") + 1;
			int L1E0_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n0->idx_u, b.n0->idx_v, "EL", "NONE") + 1;
			int E0E1_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n1->idx_u, b.n1->idx_v, "EE", "vv");
			int E1E0_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n0->idx_u, b.n0->idx_v, "EE", "vv");

			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_bend;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_bend;

			StrhBend_NC_LE[L0E1_idx] = StrhBend_NC_LE[L0E1_idx] - L0E1_bend;
		}
	}
}

void PhysicsCloth::Crimp_Bending_F(const Bend_Seg& b, Tensor& StrhBend_C_L)
{
	int L2_idx{ b.n2->idx_u + b.n2->idx_v * env->cloth->width },
		L0_idx{ b.n0->idx_u + b.n0->idx_v * env->cloth->width },
		L1_idx{ b.n1->idx_u + b.n1->idx_v * env->cloth->width };

	std::string idx_n2{ std::to_string(b.n2->idx_u) + "," + std::to_string(b.n2->idx_v) };
	std::string idx_n0{ std::to_string(b.n0->idx_u) + "," + std::to_string(b.n0->idx_v) };
	std::string idx_n1{ std::to_string(b.n1->idx_u) + "," + std::to_string(b.n1->idx_v) };

	Tensor F_x1 = ZERO31, F_x2 = ZERO31;

	if ((torch::sin(b.theta)).item<double>() != 0) {
		F_x1 = ((-2 * b.yarn.kb * b.theta) / (b.l1 * b.delta_uv * torch::sin(b.theta))) * torch::mm(b.P1, b.d2);
		F_x2 = ((-2 * b.yarn.kb * b.theta) / (b.l2 * b.delta_uv * torch::sin(b.theta))) * torch::mm(b.P2, b.d1);
	}

	Tensor F_x0 = -(F_x1 + F_x2);

	if (b.seg_type == "warp") {
		StrhBend_C_L[L2_idx] = StrhBend_C_L[L2_idx] + F_x2;
		StrhBend_C_L[L0_idx] = StrhBend_C_L[L0_idx] + F_x0;
		StrhBend_C_L[L1_idx] = StrhBend_C_L[L1_idx] + F_x1;
	}
	else {
		StrhBend_C_L[L2_idx] = StrhBend_C_L[L2_idx] - F_x2;
		StrhBend_C_L[L0_idx] = StrhBend_C_L[L0_idx] - F_x0;
		StrhBend_C_L[L1_idx] = StrhBend_C_L[L1_idx] - F_x1;
	}
}

void PhysicsCloth::Crimp_Bending_D(const Bend_Seg& b, Tensor& StrhBend_C_LE)
{
	int L2L2_idx{ find_index(b.n2->idx_u, b.n2->idx_v, b.n2->idx_u, b.n2->idx_v, "LL", "NONE") },
		L2L0_idx{ find_index(b.n2->idx_u, b.n2->idx_v, b.n0->idx_u, b.n0->idx_v, "LL", "NONE") },
		L2L1_idx{ find_index(b.n2->idx_u, b.n2->idx_v, b.n1->idx_u, b.n1->idx_v, "LL", "NONE") },
		L0L2_idx{ find_index(b.n0->idx_u, b.n0->idx_v, b.n2->idx_u, b.n2->idx_v, "LL", "NONE") },
		L0L0_idx{ find_index(b.n0->idx_u, b.n0->idx_v, b.n0->idx_u, b.n0->idx_v, "LL", "NONE") },
		L0L1_idx{ find_index(b.n0->idx_u, b.n0->idx_v, b.n1->idx_u, b.n1->idx_v, "LL", "NONE") },
		L1L2_idx{ find_index(b.n1->idx_u, b.n1->idx_v, b.n2->idx_u, b.n2->idx_v, "LL", "NONE") },
		L1L0_idx{ find_index(b.n1->idx_u, b.n1->idx_v, b.n0->idx_u, b.n0->idx_v, "LL", "NONE") },
		L1L1_idx{ find_index(b.n1->idx_u, b.n1->idx_v, b.n1->idx_u, b.n1->idx_v, "LL", "NONE") };

	Tensor L1L1_bend = ZERO33, L1L2_bend = L1L1_bend,
		L2L1_bend = L1L1_bend, L2L2_bend = L1L1_bend;
	Tensor L1E1_bend = ZERO31, L2E1_bend = ZERO31;
	Tensor E1L1_bend = ZERO13, E1L2_bend = ZERO13;

	if ((torch::sin(b.theta)).item<double>() != 0) {
		L1L1_bend = (2 * b.yarn.kb) / (pow(b.l1, 2) * b.delta_uv * torch::sin(b.theta)) *
			(b.theta * (torch::mm(torch::mm(b.P1, b.d2), b.d1.transpose(1, 0)) +
				(torch::cos(b.theta) / pow(torch::sin(b.theta), 2)) * torch::mm(torch::mm(torch::mm(b.P1, b.d2), b.d2.transpose(1, 0)), b.P1) +
				torch::cos(b.theta) * b.P1 + torch::mm(torch::mm(b.d1, b.d2.transpose(1, 0)), b.P1)) -
				(1 / torch::sin(b.theta)) * torch::mm(torch::mm(torch::mm(b.P1, b.d2), b.d2.transpose(1, 0)), b.P1));

		L1L2_bend = (-2 * b.yarn.kb) / (b.l1 * b.l2 * b.delta_uv * torch::sin(b.theta)) *
			torch::mm(b.theta * (b.P1 - (torch::cos(b.theta)) / pow(torch::sin(b.theta), 2) * torch::mm(torch::mm(b.P1, b.d2), b.d1.transpose(1, 0)))
				+ (1 / torch::sin(b.theta)) * torch::mm(torch::mm(b.P1, b.d2), b.d1.transpose(1, 0)), b.P2);

		L2L1_bend = (-2 * b.yarn.kb) / (b.l1 * b.l2 * b.delta_uv * torch::sin(b.theta)) *
			torch::mm(b.theta * (b.P2 - (torch::cos(b.theta) / pow(torch::sin(b.theta), 2)) * torch::mm(torch::mm(b.P2, b.d1), b.d2.transpose(1, 0))) +
				(1 / torch::sin(b.theta)) * torch::mm(torch::mm(b.P2, b.d1), b.d2.transpose(1, 0)), b.P1);

		L2L2_bend = (2 * b.yarn.kb) / (pow(b.l2, 2) * b.delta_uv * torch::sin(b.theta)) *
			(b.theta * (torch::mm(torch::mm(b.P2, b.d1), b.d2.transpose(1, 0)) +
				(torch::cos(b.theta) / pow(torch::sin(b.theta), 2)) * torch::mm(torch::mm(torch::mm(b.P2, b.d1), b.d1.transpose(1, 0)), b.P2) +
				torch::cos(b.theta) * b.P2 + torch::mm(torch::mm(b.d2, b.d1.transpose(1, 0)), b.P2)) -
				(1 / torch::sin(b.theta)) * torch::mm(torch::mm(torch::mm(b.P2, b.d1), b.d1.transpose(1, 0)), b.P2));

		L1E1_bend = ((2 * b.yarn.kb * b.theta) / (b.l1 * pow(b.delta_uv, 2) * torch::sin(b.theta))) * torch::mm(b.P1, b.d2);
		L2E1_bend = ((2 * b.yarn.kb * b.theta) / (b.l2 * pow(b.delta_uv, 2) * torch::sin(b.theta))) * torch::mm(b.P2, b.d1);
		E1L1_bend = ((2 * b.yarn.kb * b.theta) / (b.l1 * pow(b.delta_uv, 2) * torch::sin(b.theta))) * torch::mm(b.d2.transpose(1, 0), b.P1);
		E1L2_bend = ((2 * b.yarn.kb * b.theta) / (b.l2 * pow(b.delta_uv, 2) * torch::sin(b.theta))) * torch::mm(b.d1.transpose(1, 0), b.P2);
	}

	Tensor L1L0_bend = -(L1L1_bend + L1L2_bend);
	Tensor L2L0_bend = -(L2L1_bend + L2L2_bend);
	Tensor L0L1_bend = -(L1L1_bend + L2L1_bend);
	Tensor L0L2_bend = -(L1L2_bend + L2L2_bend);
	Tensor L0L0_bend = -(L1L0_bend + L2L0_bend);

	Tensor E1E1_bend = (-2 * b.yarn.kb * pow(b.theta, 2)) / pow(b.delta_uv, 3);
	Tensor E2E2_bend = E1E1_bend;
	Tensor E1E2_bend = -E1E1_bend, E2E1_bend = -E1E1_bend;

	Tensor L1E2_bend = -L1E1_bend;
	Tensor L2E2_bend = -L2E1_bend;

	Tensor L0E1_bend = -(L1E1_bend + L2E1_bend);
	Tensor L0E2_bend = -L0E1_bend;

	Tensor E2L1_bend = -E1L1_bend;
	Tensor E2L2_bend = -E1L2_bend;

	Tensor E1L0_bend = -(E1L1_bend + E1L2_bend);
	Tensor E2L0_bend = -E1L0_bend;

	if (!(b.n2->is_edge)) {
		if (b.seg_type == "warp") {
			int L2E2_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE");
			StrhBend_C_LE[L2E2_idx] = StrhBend_C_LE[L2E2_idx] + L2E2_bend;
		}
		else {
			int L2E2_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE") + 1;
			StrhBend_C_LE[L2E2_idx] = StrhBend_C_LE[L2E2_idx] - L2E2_bend;
		}
	}

	if (!(b.n1->is_edge)) {
		if (b.seg_type == "warp") {
			int L1E1_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE");
			StrhBend_C_LE[L1E1_idx] = StrhBend_C_LE[L1E1_idx] + L1E1_bend;
		}
		else
		{
			int L1E1_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE") + 1;
			StrhBend_C_LE[L1E1_idx] = StrhBend_C_LE[L1E1_idx] - L1E1_bend;
		}
	}

	if (!(b.n2->is_edge) && !(b.n0->is_edge)) {
		if (b.seg_type == "warp") {
			int L0E2_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE");

			StrhBend_C_LE[L0E2_idx] = StrhBend_C_LE[L0E2_idx] + L0E2_bend;

		}
		else {
			int L0E2_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE") + 1;

			StrhBend_C_LE[L0E2_idx] = StrhBend_C_LE[L0E2_idx] - L0E1_bend;
		}
	}

	if (!(b.n0->is_edge) && !(b.n1->is_edge)) {
		if (b.seg_type == "warp") {
			int L0E1_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE");

			StrhBend_C_LE[L0E1_idx] = StrhBend_C_LE[L0E1_idx] + L0E1_bend;
		}
		else {
			int L0E1_idx = find_index(b.n0->idx_u, b.n0->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE") + 1;

			StrhBend_C_LE[L0E1_idx] = StrhBend_C_LE[L0E1_idx] - L0E1_bend;
		}
	}

	if (!(b.n2->is_edge) && !(b.n1->is_edge)) {
		if (b.seg_type == "warp") {
			int L2E1_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE");
			int L1E2_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE");

			StrhBend_C_LE[L2E1_idx] = StrhBend_C_LE[L2E1_idx] + L2E1_bend;
			StrhBend_C_LE[L1E2_idx] = StrhBend_C_LE[L1E2_idx] + L1E2_bend;
		}
		else {
			int L2E1_idx = find_index(b.n2->idx_u, b.n2->idx_v, b.n1->idx_u, b.n1->idx_v, "EL", "NONE") + 1;
			int L1E2_idx = find_index(b.n1->idx_u, b.n1->idx_v, b.n2->idx_u, b.n2->idx_v, "EL", "NONE") + 1;

			StrhBend_C_LE[L2E1_idx] = StrhBend_C_LE[L2E1_idx] - L2E1_bend;
			StrhBend_C_LE[L1E2_idx] = StrhBend_C_LE[L1E2_idx] - L1E2_bend;
		}
	}
}

void PhysicsCloth::ParaColli_F(const Edge& e, Tensor& F_E)
{
	Tensor d{ ONE1 };
	if (e.n0->flip_norm != e.n1->flip_norm)
		d = 4.0 * env->cloth->R * ONE1;
	else
		d = 2.0 * env->cloth->R * ONE1;

	Tensor F_u0{ env->cloth->kc * env->cloth->L * (torch::relu(d - e.delta_uv)) };
	Tensor F_u1{ -F_u0 };

	if (!(e.n0->is_edge)) {
		int E_idx_n0{ (e.n0->idx_u - 1) + (e.n0->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp") {
			F_E[E_idx_n0] = F_E[E_idx_n0] + F_u0 * U_OneHot;
		}
		else {
			F_E[E_idx_n0] = F_E[E_idx_n0] + F_u0 * V_OneHot;
		}
	}
	if (!(e.n1->is_edge)) {
		int E_idx_n1{ (e.n1->idx_u - 1) + (e.n1->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp") {
			F_E[E_idx_n1] = F_E[E_idx_n1] + F_u1 * U_OneHot;
		}
		else {
			F_E[E_idx_n1] = F_E[E_idx_n1] + F_u1 * V_OneHot;
		}
	}
}

void PhysicsCloth::ParaColli_D(const Edge& e, Tensor& F_EE_Pos)
{
	Tensor d{ ONE1 };
	if (e.n0->flip_norm != e.n1->flip_norm)
		d = 4.0 * env->cloth->R * ONE1;
	else
		d = 2.0 * env->cloth->R * ONE1;

	Tensor E0E0_ParaColli{ (-env->cloth->kc * env->cloth->L * torch::relu(d - e.delta_uv)).view({1,1}) },
		E1E0_ParaColli{ -env->cloth->kc * env->cloth->L * torch::relu(d - e.delta_uv).view({1,1}) },
		E0E1_ParaColli{ -env->cloth->kc * env->cloth->L * torch::relu(d - e.delta_uv).view({1,1}) },
		E1E1_ParaColli{ -env->cloth->kc * env->cloth->L * torch::relu(d - e.delta_uv).view({1,1}) };

	if (!(e.n0->is_edge)) {
		if (e.edge_type == "warp") {
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };
			F_EE_Pos[E0E0_idx] = F_EE_Pos[E0E0_idx] + E0E0_ParaColli;
		}
		else {
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };
			F_EE_Pos[E0E0_idx] = F_EE_Pos[E0E0_idx] + E0E0_ParaColli;
		}
	}

	if (!(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_ParaColli;
		}
		else {
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_ParaColli;
		}
	}

	if (!(e.n0->is_edge) && !(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };
			F_EE_Pos[E0E1_idx] = F_EE_Pos[E0E1_idx] + E0E1_ParaColli;
			F_EE_Pos[E1E0_idx] = F_EE_Pos[E1E0_idx] + E1E0_ParaColli;
		}
		else {
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };
			F_EE_Pos[E0E1_idx] = F_EE_Pos[E0E1_idx] + E0E1_ParaColli;
			F_EE_Pos[E1E0_idx] = F_EE_Pos[E1E0_idx] + E1E0_ParaColli;
		}
	}
}

void PhysicsCloth::Inertia_F(const Edge& e, Tensor& F_L, Tensor& F_E)
{
	int L_idx_n0{ e.n0->idx_u + e.n0->idx_v * env->cloth->width };
	int L_idx_n1{ e.n1->idx_u + e.n1->idx_v * env->cloth->width };

	Tensor wTw{ torch::mm(torch::transpose(e.w, 0, 1), e.w) };
	Tensor w_x0{ (-1.0 / e.delta_uv) * EYE3 };
	Tensor wTw_x0{ (-2.0 / (e.delta_uv * e.delta_uv) * (e.n1->LPos - e.n0->LPos)) };

	std::vector<Tensor> w_x0_xyz{ w_x0.split(1, 1) };
	std::vector<Tensor> wTw_x0_xyz{ wTw_x0.split(1, 0) };

	Tensor w_x0_1{ w_x0_xyz[0] },
		w_x0_2{ w_x0_xyz[1] },
		w_x0_3{ w_x0_xyz[2] };

	Tensor wTw_x0_1{ wTw_x0_xyz[0] },
		wTw_x0_2{ wTw_x0_xyz[1] },
		wTw_x0_3{ wTw_x0_xyz[2] };

	Tensor M_block{
	torch::cat(
		{torch::cat({EYE3, -1 * e.w}, 1),
		torch::cat({-1 * torch::transpose(e.w,0,1), wTw}, 1)}, 0) };

	Tensor M_matrix{ torch::cat(
			{ torch::cat({2 * M_block, M_block}, 1),
			torch::cat({M_block, 2 * M_block}, 1) }, 0) };

	Tensor M_x0_1_block{
		torch::cat(
			{torch::cat({ ZERO33, -w_x0_1 }, 1),
			torch::cat({torch::transpose(-w_x0_1,0,1), wTw_x0_1}, 1) }, 0)
	};

	Tensor M_x0_2_block{
	torch::cat(
		{torch::cat({ ZERO33, -w_x0_2 }, 1),
		torch::cat({torch::transpose(-w_x0_2,0,1), wTw_x0_2}, 1) }, 0)
	};

	Tensor M_x0_3_block{
	torch::cat(
		{torch::cat({ ZERO33, -w_x0_3 }, 1),
		torch::cat({torch::transpose(-w_x0_3,0,1), wTw_x0_3}, 1) }, 0)
	};

	Tensor M_x0_1{
		1.0 / 6.0 * e.yarn.rho * e.delta_uv *
		torch::cat(
			{torch::cat({2.0 * M_x0_1_block, M_x0_1_block}, 1),
			torch::cat({M_x0_1_block, 2.0 * M_x0_1_block}, 1)}, 0) };

	Tensor M_x0_2{
	1.0 / 6.0 * e.yarn.rho * e.delta_uv *
	torch::cat(
		{torch::cat({2.0 * M_x0_2_block, M_x0_2_block}, 1),
		torch::cat({M_x0_2_block, 2.0 * M_x0_2_block}, 1)}, 0) };

	Tensor M_x0_3{
	1.0 / 6.0 * e.yarn.rho * e.delta_uv *
	torch::cat(
		{torch::cat({2.0 * M_x0_3_block, M_x0_3_block}, 1),
		torch::cat({M_x0_3_block, 2.0 * M_x0_3_block}, 1)}, 0) };

	Tensor M_x1_1{ -M_x0_1 },
		M_x1_2{ -M_x0_2 },
		M_x1_3{ -M_x0_3 };

	Tensor M_uv0_block{
	torch::cat({torch::cat({ZERO33, (-1.0 / e.delta_uv) * e.w}, 1),
		torch::cat({(-1.0 / e.delta_uv) * torch::transpose(e.w, 0, 1), (2.0 / e.delta_uv) * wTw}, 1)}, 0) };

	Tensor M_uv0{ -(1.0 / 6.0) * e.yarn.rho * M_matrix + (1.0 / 6.0) * e.yarn.rho * e.delta_uv *
		torch::cat(
			{torch::cat({2 * M_uv0_block, M_uv0_block}, 1),
			torch::cat({M_uv0_block, 2 * M_uv0_block}, 1)}, 0) };

	Tensor M_uv1{ -M_uv0 };

	Tensor T_x0_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x0_1), e.velocity_vec) };
	Tensor T_x0_2{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x0_2), e.velocity_vec) };
	Tensor T_x0_3{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x0_3), e.velocity_vec) };

	Tensor T_x1_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x1_1), e.velocity_vec) };
	Tensor T_x1_2{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x1_2), e.velocity_vec) };
	Tensor T_x1_3{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x1_3), e.velocity_vec) };

	Tensor T_x0{ torch::cat({T_x0_1, T_x0_2, T_x0_3}, 0) };
	Tensor T_x1{ torch::cat({T_x1_1, T_x1_2, T_x1_3}, 0) };

	Tensor T_uv0{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv0), e.velocity_vec) };
	Tensor T_uv1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv1), e.velocity_vec) };

	F_L[L_idx_n0] = F_L[L_idx_n0] + T_x0;
	F_L[L_idx_n1] = F_L[L_idx_n1] + T_x1;

	if (!(e.n0->is_edge)) {
		int E_idx_n0{ (e.n0->idx_u - 1) + (e.n0->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp") {
			F_E[E_idx_n0] = F_E[E_idx_n0] + T_uv0 * U_OneHot;
		}
		else {
			F_E[E_idx_n0] = F_E[E_idx_n0] + T_uv0 * V_OneHot;
		}
	}
	if (!e.n1->is_edge) {
		int E_idx_n1{ (e.n1->idx_u - 1) + (e.n1->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp") {
			F_E[E_idx_n1] = F_E[E_idx_n1] + T_uv1 * U_OneHot;
		}
		else {
			F_E[E_idx_n1] = F_E[E_idx_n1] + T_uv1 * V_OneHot;
		}
	}
}

void PhysicsCloth::Inertia_D_Pos(const Edge& e, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos)
{
	int L0L0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L1L0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L0L1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") },
		L1L1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") };

	Tensor wTw = torch::mm(torch::transpose(e.w, 0, 1), e.w);
	Tensor w_x0 = (-1.0 / e.delta_uv) * EYE3;
	Tensor wTw_x0 = (-2.0 / (e.delta_uv * e.delta_uv) * (e.n1->LPos - e.n0->LPos));

	std::vector<Tensor> w_x0_xyz = w_x0.split(1, 1);
	std::vector<Tensor> wTw_x0_xyz = wTw_x0.split(1, 0);

	Tensor w_x0_1{ w_x0_xyz[0] },
		w_x0_2{ w_x0_xyz[1] },
		w_x0_3{ w_x0_xyz[2] };

	Tensor wTw_x0_1{ wTw_x0_xyz[0] },
		wTw_x0_2{ wTw_x0_xyz[1] },
		wTw_x0_3{ wTw_x0_xyz[2] };

	Tensor M_x0_1_block{
		torch::cat(
			{torch::cat({ ZERO33, -w_x0_1 }, 1),
			torch::cat({torch::transpose(-w_x0_1,0,1), wTw_x0_1}, 1) }, 0)
	};

	Tensor M_x0_2_block{
	torch::cat(
		{torch::cat({ ZERO33, -w_x0_2 }, 1),
		torch::cat({torch::transpose(-w_x0_2,0,1), wTw_x0_2}, 1) }, 0)
	};

	Tensor M_x0_3_block{
	torch::cat(
		{torch::cat({ ZERO33, -w_x0_3 }, 1),
		torch::cat({torch::transpose(-w_x0_3,0,1), wTw_x0_3}, 1) }, 0)
	};

	Tensor M_matrix_x0_1{
		torch::cat(
			{torch::cat({2.0 * M_x0_1_block, M_x0_1_block}, 1),
			torch::cat({M_x0_1_block, 2.0 * M_x0_1_block}, 1)}, 0)
	};

	Tensor M_matrix_x0_2{
	torch::cat(
		{torch::cat({2.0 * M_x0_2_block, M_x0_2_block}, 1),
		torch::cat({M_x0_2_block, 2.0 * M_x0_2_block}, 1)}, 0)
	};

	Tensor M_matrix_x0_3{
	torch::cat(
		{torch::cat({2.0 * M_x0_3_block, M_x0_3_block}, 1),
		torch::cat({M_x0_3_block, 2.0 * M_x0_3_block}, 1)}, 0)
	};

	Tensor M_x0_1_x0_1_block{ torch::cat(
		{ torch::zeros({3,4}, opts),
		torch::cat({ZERO13, (2 / (e.delta_uv * e.delta_uv)).view({1,1})},1) }, 0) };

	Tensor M_x0_1_x0_1{ (1.0 / 6.0) * e.yarn.rho * torch::cat(
		{torch::cat({2 * M_x0_1_x0_1_block, M_x0_1_x0_1_block}, 1),
		torch::cat({M_x0_1_x0_1_block, 2 * M_x0_1_x0_1_block}, 1)},0) };

	Tensor M_x0_1_x1_1{ -M_x0_1_x0_1 };

	Tensor T_x0_1_x0_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x0_1_x0_1), e.velocity_vec) };
	Tensor T_x0_1_x0_2{ ZERO11 };
	Tensor T_x0_1_x0_3{ ZERO11 };

	Tensor T_x0_1_x1_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x0_1_x1_1), e.velocity_vec) };
	Tensor T_x0_1_x1_2{ ZERO11 };
	Tensor T_x0_1_x1_3{ ZERO11 };

	Tensor T_x0_2_x0_2{ T_x0_1_x0_1 };
	Tensor T_x0_3_x0_3{ T_x0_1_x0_1 };

	Tensor T_x0_2_x1_2{ T_x0_1_x1_1 };
	Tensor T_x0_3_x1_3{ T_x0_1_x1_1 };

	Tensor T_x0_2_x0_1{ ZERO11 };
	Tensor T_x0_2_x0_3{ ZERO11 };
	Tensor T_x0_2_x1_1{ ZERO11 };
	Tensor T_x0_2_x1_3{ ZERO11 };

	Tensor T_x0_3_x0_1{ ZERO11 };
	Tensor T_x0_3_x0_2{ ZERO11 };
	Tensor T_x0_3_x1_1{ ZERO11 };
	Tensor T_x0_3_x1_2{ ZERO11 };

	Tensor M_x1_1_x0_1{ -M_x0_1_x0_1 };
	Tensor M_x1_1_x1_1{ -M_x1_1_x0_1 };

	Tensor T_x1_1_x0_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x1_1_x0_1), e.velocity_vec) };
	Tensor T_x1_1_x0_2{ ZERO11 };
	Tensor T_x1_1_x0_3{ ZERO11 };

	Tensor T_x1_1_x1_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_x1_1_x1_1), e.velocity_vec) };
	Tensor T_x1_1_x1_2{ ZERO11 };
	Tensor T_x1_1_x1_3{ ZERO11 };

	Tensor T_x1_2_x0_2{ T_x1_1_x0_1 };
	Tensor T_x1_3_x0_3{ T_x1_1_x0_1 };

	Tensor T_x1_2_x1_2{ T_x1_1_x1_1 };
	Tensor T_x1_3_x1_3{ T_x1_1_x1_1 };

	Tensor T_x1_2_x0_1{ ZERO11 };
	Tensor T_x1_2_x0_3{ ZERO11 };
	Tensor T_x1_2_x1_1{ ZERO11 };
	Tensor T_x1_2_x1_3{ ZERO11 };

	Tensor T_x1_3_x0_1{ ZERO11 };
	Tensor T_x1_3_x0_2{ ZERO11 };
	Tensor T_x1_3_x1_1{ ZERO11 };
	Tensor T_x1_3_x1_2{ ZERO11 };

	Tensor w_x0_uv_0{ (1.0 / (e.delta_uv * e.delta_uv)) * EYE3 };
	Tensor w_x0_uv_1{ (-1.0 / (e.delta_uv * e.delta_uv)) * EYE3 };
	Tensor w_x1_uv_0{ (-1.0 / (e.delta_uv * e.delta_uv)) * EYE3 };
	Tensor w_x1_uv_1{ (1.0 / (e.delta_uv * e.delta_uv)) * EYE3 };

	Tensor wTw_x0_uv_0{ (-4.0 / (e.delta_uv * e.delta_uv)) * e.w };
	Tensor wTw_x0_uv_1{ (4.0 / (e.delta_uv * e.delta_uv)) * e.w };
	Tensor wTw_x1_uv_0{ (4.0 / (e.delta_uv * e.delta_uv)) * e.w };
	Tensor wTw_x1_uv_1{ (-4.0 / (e.delta_uv * e.delta_uv)) * e.w };

	std::vector<Tensor> w_x0_xyz_uv_0 = w_x0_uv_0.split(1, 1);
	std::vector<Tensor> w_x0_xyz_uv_1 = w_x0_uv_1.split(1, 1);
	std::vector<Tensor> w_x1_xyz_uv_0 = w_x1_uv_0.split(1, 1);
	std::vector<Tensor> w_x1_xyz_uv_1 = w_x1_uv_1.split(1, 1);

	std::vector<Tensor> wTw_x0_xyz_uv_0 = wTw_x0_uv_0.split(1, 0);
	std::vector<Tensor> wTw_x0_xyz_uv_1 = wTw_x0_uv_1.split(1, 0);
	std::vector<Tensor> wTw_x1_xyz_uv_0 = wTw_x1_uv_0.split(1, 0);
	std::vector<Tensor> wTw_x1_xyz_uv_1 = wTw_x1_uv_1.split(1, 0);

	Tensor w_x0_1_uv_0{ w_x0_xyz_uv_0[0] },
		w_x0_2_uv_0{ w_x0_xyz_uv_0[1] },
		w_x0_3_uv_0{ w_x0_xyz_uv_0[2] };

	Tensor w_x0_1_uv_1{ w_x0_xyz_uv_1[0] },
		w_x0_2_uv_1{ w_x0_xyz_uv_1[1] },
		w_x0_3_uv_1{ w_x0_xyz_uv_1[2] };

	Tensor w_x1_1_uv_0{ w_x1_xyz_uv_0[0] },
		w_x1_2_uv_0{ w_x1_xyz_uv_0[1] },
		w_x1_3_uv_0{ w_x1_xyz_uv_0[2] };

	Tensor w_x1_1_uv_1{ w_x1_xyz_uv_1[0] },
		w_x1_2_uv_1{ w_x1_xyz_uv_1[1] },
		w_x1_3_uv_1{ w_x1_xyz_uv_1[2] };

	Tensor wTw_x0_1_uv_0{ wTw_x0_xyz_uv_0[0] },
		wTw_x0_2_uv_0{ wTw_x0_xyz_uv_0[1] },
		wTw_x0_3_uv_0{ wTw_x0_xyz_uv_0[2] };

	Tensor wTw_x0_1_uv_1{ wTw_x0_xyz_uv_1[0] },
		wTw_x0_2_uv_1{ wTw_x0_xyz_uv_1[1] },
		wTw_x0_3_uv_1{ wTw_x0_xyz_uv_1[2] };

	Tensor wTw_x1_1_uv_0{ wTw_x1_xyz_uv_0[0] },
		wTw_x1_2_uv_0{ wTw_x1_xyz_uv_0[1] },
		wTw_x1_3_uv_0{ wTw_x1_xyz_uv_0[2] };

	Tensor wTw_x1_1_uv_1{ wTw_x1_xyz_uv_1[0] },
		wTw_x1_2_uv_1{ wTw_x1_xyz_uv_1[1] },
		wTw_x1_3_uv_1{ wTw_x1_xyz_uv_1[2] };

	Tensor M_matrix_x0_1_uv_0_block{
	torch::cat(
		{torch::cat({ZERO33, -w_x0_1_uv_0}, 1),
		torch::cat({-torch::transpose(-w_x0_1_uv_0, 0, 1), wTw_x0_1_uv_0}, 1)}, 0)
	};
	Tensor M_matrix_x0_1_uv_0{
		torch::cat(
			{torch::cat({2 * M_matrix_x0_1_uv_0_block, M_matrix_x0_1_uv_0_block}, 1),
			torch::cat({M_matrix_x0_1_uv_0_block, 2 * M_matrix_x0_1_uv_0_block}, 1)}, 0)
	};

	Tensor M_matrix_x0_2_uv_0_block{
		torch::cat(
			{torch::cat({ZERO33, -w_x0_2_uv_0}, 1),
			torch::cat({-torch::transpose(-w_x0_2_uv_0, 0, 1), wTw_x0_2_uv_0}, 1)}, 0)
	};
	Tensor M_matrix_x0_2_uv_0{
		torch::cat(
			{torch::cat({2 * M_matrix_x0_2_uv_0_block, M_matrix_x0_2_uv_0_block}, 1),
			torch::cat({M_matrix_x0_2_uv_0_block, 2 * M_matrix_x0_2_uv_0_block}, 1)}, 0)
	};

	Tensor M_matrix_x0_3_uv_0_block{
		torch::cat(
			{torch::cat({ZERO33, -w_x0_3_uv_0}, 1),
			torch::cat({-torch::transpose(-w_x0_3_uv_0, 0, 1), wTw_x0_3_uv_0}, 1)}, 0)
	};
	Tensor M_matrix_x0_3_uv_0{
		torch::cat(
			{torch::cat({2 * M_matrix_x0_3_uv_0_block, M_matrix_x0_3_uv_0_block}, 1),
			torch::cat({M_matrix_x0_3_uv_0_block, 2 * M_matrix_x0_3_uv_0_block}, 1)}, 0)
	};

	Tensor M_x0_1_uv_0{ (-1.0 / 6.0) * e.yarn.rho * M_matrix_x0_1 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_matrix_x0_1_uv_0 };
	Tensor M_x0_2_uv_0{ (-1.0 / 6.0) * e.yarn.rho * M_matrix_x0_2 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_matrix_x0_2_uv_0 };
	Tensor M_x0_3_uv_0{ (-1.0 / 6.0) * e.yarn.rho * M_matrix_x0_3 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_matrix_x0_3_uv_0 };

	Tensor M_x0_1_uv_1{ -M_x0_1_uv_0 };
	Tensor M_x0_2_uv_1{ -M_x0_2_uv_0 };
	Tensor M_x0_3_uv_1{ -M_x0_3_uv_0 };

	Tensor M_x1_1_uv_0{ -M_x0_1_uv_0 };
	Tensor M_x1_1_uv_1{ -M_x1_1_uv_0 };

	Tensor M_x1_2_uv_0{ -M_x0_2_uv_0 };
	Tensor M_x1_2_uv_1{ -M_x1_2_uv_0 };

	Tensor M_x1_3_uv_0{ -M_x0_3_uv_0 };
	Tensor M_x1_3_uv_1{ -M_x1_3_uv_0 };

	Tensor T_x0_1_uv_0{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x0_1_uv_0), e.velocity_vec) };
	Tensor T_x0_1_uv_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x0_1_uv_1), e.velocity_vec) };
	Tensor T_x0_2_uv_0{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x0_2_uv_0), e.velocity_vec) };
	Tensor T_x0_2_uv_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x0_2_uv_1), e.velocity_vec) };
	Tensor T_x0_3_uv_0{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x0_3_uv_0), e.velocity_vec) };
	Tensor T_x0_3_uv_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x0_3_uv_1), e.velocity_vec) };

	Tensor T_x1_1_uv_0{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x1_1_uv_0), e.velocity_vec) };
	Tensor T_x1_1_uv_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x1_1_uv_1), e.velocity_vec) };
	Tensor T_x1_2_uv_0{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x1_2_uv_0), e.velocity_vec) };
	Tensor T_x1_2_uv_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x1_2_uv_1), e.velocity_vec) };
	Tensor T_x1_3_uv_0{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x1_3_uv_0), e.velocity_vec) };
	Tensor T_x1_3_uv_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec,0,1), M_x1_3_uv_1), e.velocity_vec) };

	Tensor M_matrix_uv0_block{
	torch::cat(
		{torch::cat({ZERO33, (-1.0 / e.delta_uv) * e.w}, 1),
		torch::cat({(-1.0 / e.delta_uv) * torch::transpose(e.w, 0, 1), (2.0 / e.delta_uv) * wTw}, 1)}, 0)
	};

	Tensor M_matrix_uv_0{ torch::cat(
			{torch::cat({2 * M_matrix_uv0_block, M_matrix_uv0_block}, 1),
			torch::cat({M_matrix_uv0_block, 2 * M_matrix_uv0_block}, 1)}, 0) };

	Tensor M_matrix_uv_1{ -M_matrix_uv_0 };

	Tensor w_uv_0_uv_0{ (2 / (e.delta_uv * e.delta_uv)) * e.w };
	Tensor wTw_uv_0_uv_0{ (6 / (e.delta_uv * e.delta_uv)) * wTw };

	Tensor M_matrix_uv0_uv0_block{
		torch::cat(
			{torch::cat({ZERO33, -w_uv_0_uv_0}, 1),
			torch::cat({torch::transpose(-w_uv_0_uv_0,0,1), wTw_uv_0_uv_0}, 1)}, 0)
	};

	Tensor M_matrix_uv_0_uv_0{
		torch::cat(
			{torch::cat({2 * M_matrix_uv0_uv0_block, M_matrix_uv0_uv0_block}, 1),
			torch::cat({M_matrix_uv0_uv0_block, 2 * M_matrix_uv0_uv0_block}, 1)},0)
	};

	Tensor M_uv_0_x0_1{ (-1.0 / 6.0) * e.yarn.rho * M_matrix_x0_1 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * (-M_matrix_x0_1_uv_0) };
	Tensor M_uv_0_x0_2{ (-1.0 / 6.0) * e.yarn.rho * M_matrix_x0_2 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * (-M_matrix_x0_2_uv_0) };
	Tensor M_uv_0_x0_3{ (-1.0 / 6.0) * e.yarn.rho * M_matrix_x0_3 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * (-M_matrix_x0_3_uv_0) };

	Tensor M_uv_0_x1_1{ -M_uv_0_x0_1 };
	Tensor M_uv_0_x1_2{ -M_uv_0_x0_2 };
	Tensor M_uv_0_x1_3{ -M_uv_0_x0_3 };
	Tensor M_uv_1_x0_1{ -M_uv_0_x0_1 };
	Tensor M_uv_1_x0_2{ -M_uv_0_x0_2 };
	Tensor M_uv_1_x0_3{ -M_uv_0_x0_3 };
	Tensor M_uv_1_x1_1{ -M_uv_1_x0_1 };
	Tensor M_uv_1_x1_2{ -M_uv_1_x0_2 };
	Tensor M_uv_1_x1_3{ -M_uv_1_x0_3 };

	Tensor M_uv_0_uv_0 = (-1.0 / 6.0) * e.yarn.rho * M_matrix_uv_0 -
		(1.0 / 6.0) * e.yarn.rho * M_matrix_uv_0 +
		(1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_matrix_uv_0_uv_0;

	Tensor M_uv_0_uv_1{ -M_uv_0_uv_0 };
	Tensor M_uv_1_uv_0{ -M_uv_0_uv_0 };
	Tensor M_uv_1_uv_1{ -M_uv_1_uv_0 };

	Tensor T_uv_0_x0_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_0_x0_1), e.velocity_vec) };
	Tensor T_uv_0_x0_2{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_0_x0_2), e.velocity_vec) };
	Tensor T_uv_0_x0_3{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_0_x0_3), e.velocity_vec) };
	Tensor T_uv_0_x1_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_0_x1_1), e.velocity_vec) };
	Tensor T_uv_0_x1_2{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_0_x1_2), e.velocity_vec) };
	Tensor T_uv_0_x1_3{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_0_x1_3), e.velocity_vec) };

	Tensor T_uv_1_x0_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_1_x0_1), e.velocity_vec) };
	Tensor T_uv_1_x0_2{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_1_x0_2), e.velocity_vec) };
	Tensor T_uv_1_x0_3{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_1_x0_3), e.velocity_vec) };
	Tensor T_uv_1_x1_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_1_x1_1), e.velocity_vec) };
	Tensor T_uv_1_x1_2{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_1_x1_2), e.velocity_vec) };
	Tensor T_uv_1_x1_3{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_1_x1_3), e.velocity_vec) };

	Tensor T_uv_0_uv_0{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_0_uv_0), e.velocity_vec) };
	Tensor T_uv_0_uv_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_0_uv_1), e.velocity_vec) };
	Tensor T_uv_1_uv_0{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_1_uv_0), e.velocity_vec) };
	Tensor T_uv_1_uv_1{ 0.5 * torch::mm(torch::mm(torch::transpose(e.velocity_vec, 0, 1), M_uv_1_uv_1), e.velocity_vec) };

	Tensor L0L0_inertial{ torch::cat(
		{torch::cat({T_x0_1_x0_1, T_x0_1_x0_2, T_x0_1_x0_3},1),
		torch::cat({T_x0_2_x0_1, T_x0_2_x0_2, T_x0_2_x0_3},1),
		torch::cat({T_x0_3_x0_1, T_x0_3_x0_2, T_x0_3_x0_3},1)},0) };

	Tensor L0L1_inertial{ torch::cat(
		{torch::cat({T_x0_1_x1_1, T_x0_1_x1_2, T_x0_1_x1_3},1),
		torch::cat({T_x0_2_x1_1, T_x0_2_x1_2, T_x0_2_x1_3},1),
		torch::cat({T_x0_3_x1_1, T_x0_3_x1_2, T_x0_3_x1_3},1)},0) };;

	Tensor L1L0_inertial{ torch::cat(
		{torch::cat({T_x1_1_x0_1, T_x1_1_x0_2, T_x1_1_x0_3},1),
		torch::cat({T_x1_2_x0_1, T_x1_2_x0_2, T_x1_2_x0_3},1),
		torch::cat({T_x1_3_x0_1, T_x1_3_x0_2, T_x1_3_x0_3},1)},0) };;

	Tensor L1L1_inertial{ torch::cat(
		{torch::cat({T_x1_1_x1_1, T_x1_1_x1_2, T_x1_1_x1_3},1),
		torch::cat({T_x1_2_x1_1, T_x1_2_x1_2, T_x1_2_x1_3},1),
		torch::cat({T_x1_3_x1_1, T_x1_3_x1_2, T_x1_3_x1_3},1)},0) };;

	Tensor L0E0_inertial{ torch::cat({T_x0_1_uv_0, T_x0_2_uv_0, T_x0_3_uv_0},0) };
	Tensor L0E1_inertial{ torch::cat({T_x0_1_uv_1, T_x0_2_uv_1, T_x0_3_uv_1},0) };

	Tensor L1E0_inertial{ torch::cat({T_x1_1_uv_0, T_x1_2_uv_0, T_x1_3_uv_0},0) };
	Tensor L1E1_inertial{ torch::cat({T_x1_1_uv_1, T_x1_2_uv_1, T_x1_3_uv_1},0) };

	Tensor E0L0_inertial{ torch::cat({T_uv_0_x0_1, T_uv_0_x0_2, T_uv_0_x0_3},1) };
	Tensor E0L1_inertial{ torch::cat({T_uv_0_x1_1, T_uv_0_x1_2, T_uv_0_x1_3},1) };

	Tensor E1L0_inertial{ torch::cat({T_uv_1_x0_1, T_uv_1_x0_2, T_uv_1_x0_3},1) };
	Tensor E1L1_inertial{ torch::cat({T_uv_1_x1_1, T_uv_1_x1_2, T_uv_1_x1_3},1) };

	Tensor E0E0_inertial{ T_uv_0_uv_0 };
	Tensor E0E1_inertial{ T_uv_0_uv_1 };

	Tensor E1E0_inertial{ T_uv_1_uv_0 };
	Tensor E1E1_inertial{ T_uv_1_uv_1 };

	F_LL_Pos[L0L0_idx] = F_LL_Pos[L0L0_idx] + L0L0_inertial;
	F_LL_Pos[L0L1_idx] = F_LL_Pos[L0L1_idx] + L0L1_inertial;
	F_LL_Pos[L1L0_idx] = F_LL_Pos[L1L0_idx] + L1L0_inertial;
	F_LL_Pos[L1L1_idx] = F_LL_Pos[L1L1_idx] + L1L1_inertial;

	if (!(e.n0->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_EL_Pos[L0E0_idx] = F_EL_Pos[L0E0_idx] + E0L0_inertial;
			F_LE_Pos[L0E0_idx] = F_LE_Pos[L0E0_idx] + L0E0_inertial;
			F_EE_Pos[E0E0_idx] = F_EE_Pos[E0E0_idx] + E0E0_inertial;
		}
		else {
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };

			F_EL_Pos[L0E0_idx] = F_EL_Pos[L0E0_idx] + E0L0_inertial;
			F_LE_Pos[L0E0_idx] = F_LE_Pos[L0E0_idx] + L0E0_inertial;
			F_EE_Pos[E0E0_idx] = F_EE_Pos[E0E0_idx] + E0E0_inertial;
		}
	}

	if (!(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };

			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_inertial;
			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_inertial;
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_inertial;
		}
		else {
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1 };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };

			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_inertial;
			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_inertial;
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_inertial;
		}
	}

	if (!(e.n0->is_edge) && !(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_LE_Pos[L1E0_idx] = F_LE_Pos[L1E0_idx] + L1E0_inertial;
			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_inertial;
			F_EL_Pos[L1E0_idx] = F_EL_Pos[L1E0_idx] + E0L1_inertial;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_inertial;
			F_EE_Pos[E0E1_idx] = F_EE_Pos[E0E1_idx] + E0E1_inertial;
			F_EE_Pos[E1E0_idx] = F_EE_Pos[E1E0_idx] + E1E0_inertial;
		}
		else {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1 };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };

			F_LE_Pos[L1E0_idx] = F_LE_Pos[L1E0_idx] + L1E0_inertial;
			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_inertial;
			F_EL_Pos[L1E0_idx] = F_EL_Pos[L1E0_idx] + E0L1_inertial;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_inertial;
			F_EE_Pos[E0E1_idx] = F_EE_Pos[E0E1_idx] + E0E1_inertial;
			F_EE_Pos[E1E0_idx] = F_EE_Pos[E1E0_idx] + E1E0_inertial;
		}
	}
}

void PhysicsCloth::Inertia_D_Vel(const Edge& e, Tensor& F_LL_Vel, Tensor& F_EL_Vel, Tensor& F_LE_Vel, Tensor& F_EE_Vel)
{
	int L0L0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L1L0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L0L1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") },
		L1L1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") };

	std::vector<Tensor> vel_der = torch::eye(8, opts).split(1, 1);

	Tensor vel_x0_1{ vel_der[0] },
		vel_x0_2{ vel_der[1] },
		vel_x0_3{ vel_der[2] },
		vel_uv0{ vel_der[3] },
		vel_x1_1{ vel_der[4] },
		vel_x1_2{ vel_der[5] },
		vel_x1_3{ vel_der[6] },
		vel_uv1{ vel_der[7] };

	Tensor wTw = torch::mm(torch::transpose(e.w, 0, 1), e.w);
	Tensor w_x0 = (-1.0 / e.delta_uv) * EYE3;
	Tensor wTw_x0 = (-2.0 / (e.delta_uv * e.delta_uv) * (e.n1->LPos - e.n0->LPos));

	std::vector<Tensor> w_x0_xyz = w_x0.split(1, 1);
	std::vector<Tensor> wTw_x0_xyz = wTw_x0.split(1, 0);

	Tensor w_x0_1{ w_x0_xyz[0] },
		w_x0_2{ w_x0_xyz[1] },
		w_x0_3{ w_x0_xyz[2] };

	Tensor wTw_x0_1{ wTw_x0_xyz[0] },
		wTw_x0_2{ wTw_x0_xyz[1] },
		wTw_x0_3{ wTw_x0_xyz[2] };

	Tensor M_block{
		torch::cat(
			{torch::cat({EYE3, -1 * e.w}, 1),
			torch::cat({-1 * torch::transpose(e.w,0,1), wTw}, 1)}, 0)
	};

	Tensor M_matrix{ torch::cat(
			{ torch::cat({2 * M_block, M_block}, 1),
			torch::cat({M_block, 2 * M_block}, 1) }, 0) };

	Tensor M_x0_1_block{
		torch::cat(
			{torch::cat({ ZERO33, -w_x0_1 }, 1),
			torch::cat({torch::transpose(-w_x0_1,0,1), wTw_x0_1}, 1) }, 0)
	};

	Tensor M_x0_2_block{
	torch::cat(
		{torch::cat({ ZERO33, -w_x0_2 }, 1),
		torch::cat({torch::transpose(-w_x0_2,0,1), wTw_x0_2}, 1) }, 0)
	};

	Tensor M_x0_3_block{
	torch::cat(
		{torch::cat({ ZERO33, -w_x0_3 }, 1),
		torch::cat({torch::transpose(-w_x0_3,0,1), wTw_x0_3}, 1) }, 0)
	};

	Tensor M_x0_1{
		(1.0 / 6.0) * e.yarn.rho * e.delta_uv *
		torch::cat(
			{torch::cat({2.0 * M_x0_1_block, M_x0_1_block}, 1),
			torch::cat({M_x0_1_block, 2.0 * M_x0_1_block}, 1)}, 0)
	};

	Tensor M_x0_2{
		(1.0 / 6.0) * e.yarn.rho * e.delta_uv *
		torch::cat(
			{torch::cat({2.0 * M_x0_2_block, M_x0_2_block}, 1),
			torch::cat({M_x0_2_block, 2.0 * M_x0_2_block}, 1)}, 0)
	};

	Tensor M_x0_3{
		(1.0 / 6.0) * e.yarn.rho * e.delta_uv *
		torch::cat(
			{torch::cat({2.0 * M_x0_3_block, M_x0_3_block}, 1),
			torch::cat({M_x0_3_block, 2.0 * M_x0_3_block}, 1)}, 0)
	};

	Tensor M_x1_1{ -M_x0_1 };
	Tensor M_x1_2{ -M_x0_2 };
	Tensor M_x1_3{ -M_x0_3 };

	Tensor M_uv0_block{
	torch::cat(
		{torch::cat({ZERO33, (-1.0 / e.delta_uv) * e.w}, 1),
		torch::cat({(-1.0 / e.delta_uv) * torch::transpose(e.w, 0, 1), (2.0 / e.delta_uv) * wTw}, 1)}, 0)
	};

	Tensor M_uv0{ -(1.0 / 6.0) * e.yarn.rho * M_matrix + (1.0 / 6.0) * e.yarn.rho * e.delta_uv *
		torch::cat(
			{torch::cat({2 * M_uv0_block, M_uv0_block}, 1),
			torch::cat({M_uv0_block, 2 * M_uv0_block}, 1)}, 0) };

	Tensor M_uv1{ -M_uv0 };

	Tensor T_x0_1_x0_1{ torch::mm(torch::mm(torch::transpose(vel_x0_1, 0, 1), M_x0_1), e.velocity_vec) };
	Tensor T_x0_1_x0_2{ torch::mm(torch::mm(torch::transpose(vel_x0_2, 0, 1), M_x0_1), e.velocity_vec) };
	Tensor T_x0_1_x0_3{ torch::mm(torch::mm(torch::transpose(vel_x0_3, 0, 1), M_x0_1), e.velocity_vec) };

	Tensor T_x0_1_x1_1{ torch::mm(torch::mm(torch::transpose(vel_x1_1, 0, 1), M_x0_1), e.velocity_vec) };
	Tensor T_x0_1_x1_2{ torch::mm(torch::mm(torch::transpose(vel_x1_2, 0, 1), M_x0_1), e.velocity_vec) };
	Tensor T_x0_1_x1_3{ torch::mm(torch::mm(torch::transpose(vel_x1_3, 0, 1), M_x0_1), e.velocity_vec) };

	Tensor T_x0_1_uv0{ torch::mm(torch::mm(torch::transpose(vel_uv0, 0, 1), M_x0_1), e.velocity_vec) };
	Tensor T_x0_1_uv1{ torch::mm(torch::mm(torch::transpose(vel_uv1, 0, 1), M_x0_1), e.velocity_vec) };

	Tensor T_x0_2_x0_1{ torch::mm(torch::mm(torch::transpose(vel_x0_1, 0, 1), M_x0_2), e.velocity_vec) };
	Tensor T_x0_2_x0_2{ torch::mm(torch::mm(torch::transpose(vel_x0_2, 0, 1), M_x0_2), e.velocity_vec) };
	Tensor T_x0_2_x0_3{ torch::mm(torch::mm(torch::transpose(vel_x0_3, 0, 1), M_x0_2), e.velocity_vec) };

	Tensor T_x0_2_x1_1{ torch::mm(torch::mm(torch::transpose(vel_x1_1, 0, 1), M_x0_2), e.velocity_vec) };
	Tensor T_x0_2_x1_2{ torch::mm(torch::mm(torch::transpose(vel_x1_2, 0, 1), M_x0_2), e.velocity_vec) };
	Tensor T_x0_2_x1_3{ torch::mm(torch::mm(torch::transpose(vel_x1_3, 0, 1), M_x0_2), e.velocity_vec) };

	Tensor T_x0_2_uv0{ torch::mm(torch::mm(torch::transpose(vel_uv0, 0, 1), M_x0_2), e.velocity_vec) };
	Tensor T_x0_2_uv1{ torch::mm(torch::mm(torch::transpose(vel_uv1, 0, 1), M_x0_2), e.velocity_vec) };

	Tensor T_x0_3_x0_1{ torch::mm(torch::mm(torch::transpose(vel_x0_1, 0, 1), M_x0_3), e.velocity_vec) };
	Tensor T_x0_3_x0_2{ torch::mm(torch::mm(torch::transpose(vel_x0_2, 0, 1), M_x0_3), e.velocity_vec) };
	Tensor T_x0_3_x0_3{ torch::mm(torch::mm(torch::transpose(vel_x0_3, 0, 1), M_x0_3), e.velocity_vec) };

	Tensor T_x0_3_x1_1{ torch::mm(torch::mm(torch::transpose(vel_x1_1, 0, 1), M_x0_3), e.velocity_vec) };
	Tensor T_x0_3_x1_2{ torch::mm(torch::mm(torch::transpose(vel_x1_2, 0, 1), M_x0_3), e.velocity_vec) };
	Tensor T_x0_3_x1_3{ torch::mm(torch::mm(torch::transpose(vel_x1_3, 0, 1), M_x0_3), e.velocity_vec) };

	Tensor T_x0_3_uv0{ torch::mm(torch::mm(torch::transpose(vel_uv0, 0, 1), M_x0_3), e.velocity_vec) };
	Tensor T_x0_3_uv1{ torch::mm(torch::mm(torch::transpose(vel_uv1, 0, 1), M_x0_3), e.velocity_vec) };

	Tensor T_x1_1_x0_1{ torch::mm(torch::mm(torch::transpose(vel_x0_1, 0, 1), M_x1_1), e.velocity_vec) };
	Tensor T_x1_1_x0_2{ torch::mm(torch::mm(torch::transpose(vel_x0_2, 0, 1), M_x1_1), e.velocity_vec) };
	Tensor T_x1_1_x0_3{ torch::mm(torch::mm(torch::transpose(vel_x0_3, 0, 1), M_x1_1), e.velocity_vec) };

	Tensor T_x1_1_x1_1{ torch::mm(torch::mm(torch::transpose(vel_x1_1, 0, 1), M_x1_1), e.velocity_vec) };
	Tensor T_x1_1_x1_2{ torch::mm(torch::mm(torch::transpose(vel_x1_2, 0, 1), M_x1_1), e.velocity_vec) };
	Tensor T_x1_1_x1_3{ torch::mm(torch::mm(torch::transpose(vel_x1_3, 0, 1), M_x1_1), e.velocity_vec) };

	Tensor T_x1_1_uv0{ torch::mm(torch::mm(torch::transpose(vel_uv0, 0, 1), M_x1_1), e.velocity_vec) };
	Tensor T_x1_1_uv1{ torch::mm(torch::mm(torch::transpose(vel_uv1, 0, 1), M_x1_1), e.velocity_vec) };

	Tensor T_x1_2_x0_1{ torch::mm(torch::mm(torch::transpose(vel_x0_1, 0, 1), M_x1_2), e.velocity_vec) };
	Tensor T_x1_2_x0_2{ torch::mm(torch::mm(torch::transpose(vel_x0_2, 0, 1), M_x1_2), e.velocity_vec) };
	Tensor T_x1_2_x0_3{ torch::mm(torch::mm(torch::transpose(vel_x0_3, 0, 1), M_x1_2), e.velocity_vec) };

	Tensor T_x1_2_x1_1{ torch::mm(torch::mm(torch::transpose(vel_x1_1, 0, 1), M_x1_2), e.velocity_vec) };
	Tensor T_x1_2_x1_2{ torch::mm(torch::mm(torch::transpose(vel_x1_2, 0, 1), M_x1_2), e.velocity_vec) };
	Tensor T_x1_2_x1_3{ torch::mm(torch::mm(torch::transpose(vel_x1_3, 0, 1), M_x1_2), e.velocity_vec) };

	Tensor T_x1_2_uv0{ torch::mm(torch::mm(torch::transpose(vel_uv0, 0, 1), M_x1_2), e.velocity_vec) };
	Tensor T_x1_2_uv1{ torch::mm(torch::mm(torch::transpose(vel_uv1, 0, 1), M_x1_2), e.velocity_vec) };

	Tensor T_x1_3_x0_1{ torch::mm(torch::mm(torch::transpose(vel_x0_1, 0, 1), M_x1_3), e.velocity_vec) };
	Tensor T_x1_3_x0_2{ torch::mm(torch::mm(torch::transpose(vel_x0_2, 0, 1), M_x1_3), e.velocity_vec) };
	Tensor T_x1_3_x0_3{ torch::mm(torch::mm(torch::transpose(vel_x0_3, 0, 1), M_x1_3), e.velocity_vec) };

	Tensor T_x1_3_x1_1{ torch::mm(torch::mm(torch::transpose(vel_x1_1, 0, 1), M_x1_3), e.velocity_vec) };
	Tensor T_x1_3_x1_2{ torch::mm(torch::mm(torch::transpose(vel_x1_2, 0, 1), M_x1_3), e.velocity_vec) };
	Tensor T_x1_3_x1_3{ torch::mm(torch::mm(torch::transpose(vel_x1_3, 0, 1), M_x1_3), e.velocity_vec) };

	Tensor T_x1_3_uv0{ torch::mm(torch::mm(torch::transpose(vel_uv0, 0, 1), M_x1_3), e.velocity_vec) };
	Tensor T_x1_3_uv1{ torch::mm(torch::mm(torch::transpose(vel_uv1, 0, 1), M_x1_3), e.velocity_vec) };

	Tensor T_uv0_x0_1{ torch::mm(torch::mm(torch::transpose(vel_x0_1, 0, 1), M_uv0), e.velocity_vec) };
	Tensor T_uv0_x0_2{ torch::mm(torch::mm(torch::transpose(vel_x0_2, 0, 1), M_uv0), e.velocity_vec) };
	Tensor T_uv0_x0_3{ torch::mm(torch::mm(torch::transpose(vel_x0_3, 0, 1), M_uv0), e.velocity_vec) };

	Tensor T_uv0_uv0{ torch::mm(torch::mm(torch::transpose(vel_uv0, 0, 1), M_uv0), e.velocity_vec) };

	Tensor T_uv0_x1_1{ torch::mm(torch::mm(torch::transpose(vel_x1_1, 0, 1), M_uv0), e.velocity_vec) };
	Tensor T_uv0_x1_2{ torch::mm(torch::mm(torch::transpose(vel_x1_2, 0, 1), M_uv0), e.velocity_vec) };
	Tensor T_uv0_x1_3{ torch::mm(torch::mm(torch::transpose(vel_x1_3, 0, 1), M_uv0), e.velocity_vec) };

	Tensor T_uv0_uv1{ torch::mm(torch::mm(torch::transpose(vel_uv1, 0, 1), M_uv0), e.velocity_vec) };

	Tensor T_uv1_x0_1{ torch::mm(torch::mm(torch::transpose(vel_x0_1, 0, 1), M_uv1), e.velocity_vec) };
	Tensor T_uv1_x0_2{ torch::mm(torch::mm(torch::transpose(vel_x0_2, 0, 1), M_uv1), e.velocity_vec) };
	Tensor T_uv1_x0_3{ torch::mm(torch::mm(torch::transpose(vel_x0_3, 0, 1), M_uv1), e.velocity_vec) };

	Tensor T_uv1_uv0{ torch::mm(torch::mm(torch::transpose(vel_uv0, 0, 1), M_uv1), e.velocity_vec) };

	Tensor T_uv1_x1_1{ torch::mm(torch::mm(torch::transpose(vel_x1_1, 0, 1), M_uv1), e.velocity_vec) };
	Tensor T_uv1_x1_2{ torch::mm(torch::mm(torch::transpose(vel_x1_2, 0, 1), M_uv1), e.velocity_vec) };
	Tensor T_uv1_x1_3{ torch::mm(torch::mm(torch::transpose(vel_x1_3, 0, 1), M_uv1), e.velocity_vec) };

	Tensor T_uv1_uv1{ torch::mm(torch::mm(torch::transpose(vel_uv1, 0, 1), M_uv1), e.velocity_vec) };

	Tensor L0L0_Inertial_Vel{ torch::cat(
		{torch::cat({T_x0_1_x0_1, T_x0_1_x0_2, T_x0_1_x0_3}, 1),
		torch::cat({T_x0_2_x0_1, T_x0_2_x0_2, T_x0_2_x0_3}, 1),
		torch::cat({T_x0_3_x0_1, T_x0_3_x0_2, T_x0_3_x0_3}, 1)}, 0) };

	Tensor L0L1_Inertial_Vel{ torch::cat(
		{torch::cat({T_x0_1_x1_1, T_x0_1_x1_2, T_x0_1_x1_3}, 1),
		torch::cat({T_x0_2_x1_1, T_x0_2_x1_2, T_x0_2_x1_3}, 1),
		torch::cat({T_x0_3_x1_1, T_x0_3_x1_2, T_x0_3_x1_3}, 1)}, 0) };

	Tensor L1L0_Inertial_Vel{ torch::cat(
		{torch::cat({T_x1_1_x0_1, T_x1_1_x0_2, T_x1_1_x0_3}, 1),
		torch::cat({T_x1_2_x0_1, T_x1_2_x0_2, T_x1_2_x0_3}, 1),
		torch::cat({T_x1_3_x0_1, T_x1_3_x0_2, T_x1_3_x0_3}, 1)}, 0) };

	Tensor L1L1_Inertial_Vel{ torch::cat(
		{torch::cat({T_x1_1_x1_1, T_x1_1_x1_2, T_x1_1_x1_3}, 1),
		torch::cat({T_x1_2_x1_1, T_x1_2_x1_2, T_x1_2_x1_3}, 1),
		torch::cat({T_x1_3_x1_1, T_x1_3_x1_2, T_x1_3_x1_3}, 1)}, 0) };

	Tensor L0E0_Inertial_Vel{ torch::cat({T_x0_1_uv0, T_x0_2_uv0, T_x0_3_uv0},0) };
	Tensor L0E1_Inertial_Vel{ torch::cat({T_x0_1_uv1, T_x0_2_uv1, T_x0_3_uv1},0) };

	Tensor L1E0_Inertial_Vel{ torch::cat({T_x1_1_uv0, T_x1_2_uv0, T_x1_3_uv0},0) };
	Tensor L1E1_Inertial_Vel{ torch::cat({T_x1_1_uv1, T_x1_2_uv1, T_x1_3_uv1},0) };

	Tensor E0L0_Inertial_Vel{ torch::cat({T_uv0_x0_1, T_uv0_x0_2, T_uv0_x0_3},1) };
	Tensor E0L1_Inertial_Vel{ torch::cat({T_uv0_x1_1, T_uv0_x1_2, T_uv0_x1_3},1) };

	Tensor E1L0_Inertial_Vel{ torch::cat({T_uv1_x0_1, T_uv1_x0_2, T_uv1_x0_3},1) };
	Tensor E1L1_Inertial_Vel{ torch::cat({T_uv1_x1_1, T_uv1_x1_2, T_uv1_x1_3},1) };

	Tensor E0E0_Inertial_Vel{ T_uv0_uv0 };
	Tensor E0E1_Inertial_Vel{ T_uv0_uv1 };

	Tensor E1E0_Inertial_Vel{ T_uv1_uv0 };
	Tensor E1E1_Inertial_Vel{ T_uv1_uv1 };

	F_LL_Vel[L0L0_idx] = F_LL_Vel[L0L0_idx] + L0L0_Inertial_Vel;
	F_LL_Vel[L0L1_idx] = F_LL_Vel[L0L1_idx] + L0L1_Inertial_Vel;
	F_LL_Vel[L1L0_idx] = F_LL_Vel[L1L0_idx] + L1L0_Inertial_Vel;
	F_LL_Vel[L1L1_idx] = F_LL_Vel[L1L1_idx] + L1L1_Inertial_Vel;

	if (!(e.n0->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_EL_Vel[L0E0_idx] = F_EL_Vel[L0E0_idx] + E0L0_Inertial_Vel;
			F_LE_Vel[L0E0_idx] = F_LE_Vel[L0E0_idx] + L0E0_Inertial_Vel;
			F_EE_Vel[E0E0_idx] = F_EE_Vel[E0E0_idx] + E0E0_Inertial_Vel;
		}
		else {
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };

			F_EL_Vel[L0E0_idx] = F_EL_Vel[L0E0_idx] + E0L0_Inertial_Vel;
			F_LE_Vel[L0E0_idx] = F_LE_Vel[L0E0_idx] + L0E0_Inertial_Vel;
			F_EE_Vel[E0E0_idx] = F_EE_Vel[E0E0_idx] + E0E0_Inertial_Vel;
		}
	}

	if (!(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };

			F_EL_Vel[L1E1_idx] = F_EL_Vel[L1E1_idx] + E1L1_Inertial_Vel;
			F_LE_Vel[L1E1_idx] = F_LE_Vel[L1E1_idx] + L1E1_Inertial_Vel;
			F_EE_Vel[E1E1_idx] = F_EE_Vel[E1E1_idx] + E1E1_Inertial_Vel;
		}
		else
		{
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1 };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };

			F_EL_Vel[L1E1_idx] = F_EL_Vel[L1E1_idx] + E1L1_Inertial_Vel;
			F_LE_Vel[L1E1_idx] = F_LE_Vel[L1E1_idx] + L1E1_Inertial_Vel;
			F_EE_Vel[E1E1_idx] = F_EE_Vel[E1E1_idx] + E1E1_Inertial_Vel;
		}
	}

	if (!(e.n0->is_edge) && !(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_LE_Vel[L1E0_idx] = F_LE_Vel[L1E0_idx] + L1E0_Inertial_Vel;
			F_LE_Vel[L0E1_idx] = F_LE_Vel[L0E1_idx] + L0E1_Inertial_Vel;
			F_EL_Vel[L1E0_idx] = F_EL_Vel[L1E0_idx] + E0L1_Inertial_Vel;
			F_EL_Vel[L0E1_idx] = F_EL_Vel[L0E1_idx] + E1L0_Inertial_Vel;
			F_EE_Vel[E0E1_idx] = F_EE_Vel[E0E1_idx] + E0E1_Inertial_Vel;
			F_EE_Vel[E1E0_idx] = F_EE_Vel[E1E0_idx] + E1E0_Inertial_Vel;
		}
		else {
			int L0E1_idx = find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1;
			int L1E0_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1;
			int E0E1_idx = find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv");
			int E1E0_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv");

			F_LE_Vel[L1E0_idx] = F_LE_Vel[L1E0_idx] + L1E0_Inertial_Vel;
			F_LE_Vel[L0E1_idx] = F_LE_Vel[L0E1_idx] + L0E1_Inertial_Vel;
			F_EL_Vel[L1E0_idx] = F_EL_Vel[L1E0_idx] + E0L1_Inertial_Vel;
			F_EL_Vel[L0E1_idx] = F_EL_Vel[L0E1_idx] + E1L0_Inertial_Vel;
			F_EE_Vel[E0E1_idx] = F_EE_Vel[E0E1_idx] + E0E1_Inertial_Vel;
			F_EE_Vel[E1E0_idx] = F_EE_Vel[E1E0_idx] + E1E0_Inertial_Vel;
		}
	}
}

void PhysicsCloth::MDotVDot_F(const Edge& e, Tensor& F_L, Tensor& F_E)
{
	int L_idx_n0{ e.n0->idx_u + e.n0->idx_v * env->cloth->width };
	int L_idx_n1{ e.n1->idx_u + e.n1->idx_v * env->cloth->width };

	Tensor delta_uv_dot;

	if (e.edge_type == "warp") {
		delta_uv_dot = e.n1->EVel[0] - e.n0->EVel[0];
	}
	else {
		delta_uv_dot = e.n1->EVel[1] - e.n0->EVel[1];
	}

	Tensor wTw{ torch::mm(torch::transpose(e.w, 0, 1), e.w) };
	Tensor w_dot{ ((e.n1->LVel - e.n0->LVel) * e.delta_uv - (e.n1->LPos - e.n0->LPos) * delta_uv_dot) / (e.delta_uv * e.delta_uv) };
	Tensor wTw_dot{ torch::mm(torch::transpose(e.w, 0, 1), w_dot) + torch::mm(torch::transpose(w_dot, 0, 1), e.w) };

	Tensor M_block{ torch::cat(
		{torch::cat({EYE3, -e.w}, 1),
		torch::cat({torch::transpose(-e.w,0,1), wTw}, 1)}, 0) };

	Tensor M_matrix{ torch::cat(
			{ torch::cat({2 * M_block, M_block}, 1),
			torch::cat({M_block, 2 * M_block}, 1) }, 0) };

	Tensor M_dot_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot}, 1),
		torch::cat({torch::transpose(-w_dot, 0, 1), wTw_dot}, 1)}, 0) };

	Tensor M_dot{ (1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix + (1.0 / 6.0) * e.yarn.rho * e.delta_uv *
		torch::cat({torch::cat({2 * M_dot_block, M_dot_block},1),
		torch::cat({M_dot_block, 2 * M_dot_block},1)}, 0) };

	Tensor M_dot_V_dot{ torch::mm(M_dot, e.velocity_vec) };

	F_L[L_idx_n0] = F_L[L_idx_n0] - M_dot_V_dot.index({ Slice(0, 3) });
	F_L[L_idx_n1] = F_L[L_idx_n1] - M_dot_V_dot.index({ Slice(4, 7) });

	if (!(e.n0->is_edge)) {
		int E_idx_n0{ (e.n0->idx_u - 1) + (e.n0->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp") {
			F_E[E_idx_n0] = F_E[E_idx_n0] - M_dot_V_dot.index({ 3 }) * U_OneHot;
		}
		else {
			F_E[E_idx_n0] = F_E[E_idx_n0] - M_dot_V_dot.index({ 3 }) * V_OneHot;
		}
	}

	if (!e.n1->is_edge) {
		int E_idx_n1{ (e.n1->idx_u - 1) + (e.n1->idx_v - 1) * (env->cloth->width - 2) };
		if (e.edge_type == "warp") {
			F_E[E_idx_n1] = F_E[E_idx_n1] - M_dot_V_dot.index({ 7 }) * U_OneHot;
		}
		else {
			F_E[E_idx_n1] = F_E[E_idx_n1] - M_dot_V_dot.index({ 7 }) * V_OneHot;
		}
	}
}

void PhysicsCloth::MDotVDot_D_Pos(const Edge& e, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos)
{

	int L0L0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L1L0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L0L1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") },
		L1L1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") };

	Tensor delta_uv_dot;

	if (e.edge_type == "warp") {
		delta_uv_dot = e.n1->EVel[0] - e.n0->EVel[0];
	}
	else {
		delta_uv_dot = e.n1->EVel[1] - e.n0->EVel[1];
	}

	Tensor wTw{ torch::mm(torch::transpose(e.w, 0, 1), e.w) };
	Tensor w_dot{ ((e.n1->LVel - e.n0->LVel) * e.delta_uv - (e.n1->LPos - e.n0->LPos) * delta_uv_dot) / (e.delta_uv * e.delta_uv) };
	Tensor wTw_dot{ torch::mm(torch::transpose(e.w, 0, 1), w_dot) + torch::mm(torch::transpose(w_dot, 0, 1), e.w) };

	Tensor M_block{ torch::cat(
		{torch::cat({EYE3, -e.w}, 1),
		torch::cat({torch::transpose(-e.w,0,1), wTw}, 1)}, 0)
	};

	Tensor M_matrix{ torch::cat({ torch::cat({2 * M_block, M_block}, 1),
			torch::cat({M_block, 2 * M_block}, 1) }, 0) };

	Tensor M_dot_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot}, 1),
		torch::cat({torch::transpose(-w_dot, 0, 1), wTw_dot}, 1)}, 0) };

	Tensor M_dot{ (1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix + (1.0 / 6.0) * e.yarn.rho * e.delta_uv *
		torch::cat({torch::cat({2 * M_dot_block, M_dot_block},1),
		torch::cat({M_dot_block, 2 * M_dot_block},1)}, 0) };

	Tensor w_x0{ (-1.0 / e.delta_uv) * EYE3 };
	Tensor w_x1{ (1.0 / e.delta_uv) * EYE3 };
	Tensor wTw_x0{ (-2.0 / (e.delta_uv * e.delta_uv) * (e.n1->LPos - e.n0->LPos)) };
	Tensor wTw_x1{ (2.0 / (e.delta_uv * e.delta_uv) * (e.n1->LPos - e.n0->LPos)) };
	Tensor w_dot_x0{ (delta_uv_dot / (e.delta_uv * e.delta_uv)) * EYE3 };
	Tensor w_dot_x1{ -(delta_uv_dot / (e.delta_uv * e.delta_uv)) * EYE3 };

	std::vector<Tensor> w_x0_xyz = w_x0.split(1, 1);
	std::vector<Tensor> w_x1_xyz = w_x1.split(1, 1);
	std::vector<Tensor> wTw_x0_xyz = wTw_x0.split(1, 0);
	std::vector<Tensor> wTw_x1_xyz = wTw_x1.split(1, 0);
	std::vector<Tensor> w_dot_x0_xyz = w_dot_x0.split(1, 1);
	std::vector<Tensor> w_dot_x1_xyz = w_dot_x1.split(1, 1);

	Tensor w_x0_1{ w_x0_xyz[0] },
		w_x0_2{ w_x0_xyz[1] },
		w_x0_3{ w_x0_xyz[2] };

	Tensor w_x1_1{ w_x1_xyz[0] },
		w_x1_2{ w_x1_xyz[1] },
		w_x1_3{ w_x1_xyz[2] };

	Tensor wTw_x0_1{ wTw_x0_xyz[0] },
		wTw_x0_2{ wTw_x0_xyz[1] },
		wTw_x0_3{ wTw_x0_xyz[2] };

	Tensor wTw_x1_1{ wTw_x1_xyz[0] },
		wTw_x1_2{ wTw_x1_xyz[1] },
		wTw_x1_3{ wTw_x1_xyz[2] };

	Tensor w_dot_x0_1{ w_dot_x0_xyz[0] },
		w_dot_x0_2{ w_dot_x0_xyz[1] },
		w_dot_x0_3{ w_dot_x0_xyz[2] };

	Tensor w_dot_x1_1{ w_dot_x1_xyz[0] },
		w_dot_x1_2{ w_dot_x1_xyz[1] },
		w_dot_x1_3{ w_dot_x1_xyz[2] };

	Tensor wTw_dot_x0_1 = 2 * torch::mm(torch::transpose(w_x0_1, 0, 1), w_dot) + 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_x0_1);
	Tensor wTw_dot_x0_2 = 2 * torch::mm(torch::transpose(w_x0_2, 0, 1), w_dot) + 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_x0_2);
	Tensor wTw_dot_x0_3 = 2 * torch::mm(torch::transpose(w_x0_3, 0, 1), w_dot) + 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_x0_3);

	Tensor wTw_dot_x1_1 = 2 * torch::mm(torch::transpose(w_x1_1, 0, 1), w_dot) + 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_x1_1);
	Tensor wTw_dot_x1_2 = 2 * torch::mm(torch::transpose(w_x1_2, 0, 1), w_dot) + 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_x1_2);
	Tensor wTw_dot_x1_3 = 2 * torch::mm(torch::transpose(w_x1_3, 0, 1), w_dot) + 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_x1_3);

	Tensor w_uv0 = { -e.w / e.delta_uv },
		w_uv1{ e.w / e.delta_uv };

	Tensor wTw_uv0{ -(2 / e.delta_uv) * torch::mm(torch::transpose(e.w,0,1), e.w) },
		wTw_uv1{ (2 / e.delta_uv) * torch::mm(torch::transpose(e.w, 0, 1), e.w) };

	Tensor w_dot_uv0{ (e.n1->LVel - e.n0->LVel) / (e.delta_uv * e.delta_uv) -
		(4 * (e.n1->LPos - e.n0->LPos) * delta_uv_dot) / (e.delta_uv * e.delta_uv * e.delta_uv) };
	Tensor w_dot_uv1{ -w_dot_uv0 };

	Tensor wTw_dot_uv0{ 2 * torch::mm(torch::transpose(w_uv0, 0, 1), w_dot) + 2 * torch::mm(torch::transpose(w_dot_uv0, 0, 1), e.w) };
	Tensor wTw_dot_uv1{ 2 * torch::mm(torch::transpose(w_uv1, 0, 1), w_dot) + 2 * torch::mm(torch::transpose(w_dot_uv1, 0, 1), e.w) };

	Tensor M_matrix_x0_1_block{
	torch::cat({torch::cat({ ZERO33, -w_x0_1 }, 1),
		torch::cat({torch::transpose(-w_x0_1, 0, 1), wTw_x0_1}, 1) }, 0) };

	Tensor M_matrix_x0_2_block{
	torch::cat({torch::cat({ ZERO33, -w_x0_2 }, 1),
		torch::cat({torch::transpose(-w_x0_2, 0, 1), wTw_x0_2}, 1) }, 0) };

	Tensor M_matrix_x0_3_block{
	torch::cat({torch::cat({ ZERO33, -w_x0_3 }, 1),
		torch::cat({torch::transpose(-w_x0_3, 0, 1), wTw_x0_3}, 1) }, 0) };

	Tensor M_matrix_x0_1{
		torch::cat({torch::cat({2.0 * M_matrix_x0_1_block, M_matrix_x0_1_block}, 1),
			torch::cat({M_matrix_x0_1_block, 2.0 * M_matrix_x0_1_block}, 1)}, 0) };

	Tensor M_matrix_x0_2{
	torch::cat({torch::cat({2.0 * M_matrix_x0_2_block, M_matrix_x0_2_block}, 1),
		torch::cat({M_matrix_x0_2_block, 2.0 * M_matrix_x0_2_block}, 1)}, 0) };

	Tensor M_matrix_x0_3{
	torch::cat({torch::cat({2.0 * M_matrix_x0_3_block, M_matrix_x0_3_block}, 1),
		torch::cat({M_matrix_x0_3_block, 2.0 * M_matrix_x0_3_block}, 1)}, 0) };

	Tensor M_dot_matrix_x0_1_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot_x0_1}, 1),
		torch::cat({torch::transpose(-w_dot_x0_1, 0, 1), wTw_dot_x0_1}, 1)}, 0) };

	Tensor M_dot_matrix_x0_2_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot_x0_2}, 1),
		torch::cat({torch::transpose(-w_dot_x0_2, 0, 1), wTw_dot_x0_2}, 1)}, 0) };

	Tensor M_dot_matrix_x0_3_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot_x0_3}, 1),
		torch::cat({torch::transpose(-w_dot_x0_3, 0, 1), wTw_dot_x0_3}, 1)}, 0) };

	Tensor M_dot_matrix_x0_1{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x0_1_block, M_dot_matrix_x0_1_block}, 1),
		torch::cat({M_dot_matrix_x0_1_block, 2 * M_dot_matrix_x0_1_block}, 1)}, 0) };

	Tensor M_dot_matrix_x0_2{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x0_2_block, M_dot_matrix_x0_2_block}, 1),
		torch::cat({M_dot_matrix_x0_2_block, 2 * M_dot_matrix_x0_2_block}, 1)}, 0) };

	Tensor M_dot_matrix_x0_3{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x0_3_block, M_dot_matrix_x0_3_block}, 1),
		torch::cat({M_dot_matrix_x0_3_block, 2 * M_dot_matrix_x0_3_block}, 1)}, 0) };

	Tensor M_matrix_x1_1_block{
		torch::cat({torch::cat({ ZERO33, -w_x1_1 }, 1),
		torch::cat({torch::transpose(-w_x1_1, 0, 1), wTw_x1_1}, 1) }, 0) };

	Tensor M_matrix_x1_2_block{
		torch::cat({torch::cat({ ZERO33, -w_x1_2 }, 1),
		torch::cat({torch::transpose(-w_x1_2, 0, 1), wTw_x1_2}, 1) }, 0) };

	Tensor M_matrix_x1_3_block{
		torch::cat({torch::cat({ ZERO33, -w_x1_3 }, 1),
		torch::cat({torch::transpose(-w_x1_3, 0, 1), wTw_x1_3}, 1) }, 0) };

	Tensor M_matrix_x1_1{
		torch::cat({torch::cat({2 * M_matrix_x1_1_block, M_matrix_x1_1_block}, 1),
			torch::cat({M_matrix_x1_1_block, 2 * M_matrix_x1_1_block}, 1)}, 0) };

	Tensor M_matrix_x1_2{
	torch::cat({torch::cat({2 * M_matrix_x1_2_block, M_matrix_x1_2_block}, 1),
		torch::cat({M_matrix_x1_2_block, 2 * M_matrix_x1_2_block}, 1)}, 0) };

	Tensor M_matrix_x1_3{
	torch::cat({torch::cat({2 * M_matrix_x1_3_block, M_matrix_x1_3_block}, 1),
		torch::cat({M_matrix_x1_3_block, 2 * M_matrix_x1_3_block}, 1)}, 0) };

	Tensor M_dot_matrix_x1_1_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot_x1_1}, 1),
		torch::cat({torch::transpose(-w_dot_x1_1, 0, 1), wTw_dot_x1_1}, 1)}, 0) };

	Tensor M_dot_matrix_x1_2_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot_x1_2}, 1),
		torch::cat({torch::transpose(-w_dot_x1_2, 0, 1), wTw_dot_x1_2}, 1)}, 0) };

	Tensor M_dot_matrix_x1_3_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot_x1_3}, 1),
		torch::cat({torch::transpose(-w_dot_x1_3, 0, 1), wTw_dot_x1_3}, 1)}, 0) };

	Tensor M_dot_matrix_x1_1{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x1_1_block, M_dot_matrix_x1_1_block}, 1),
		torch::cat({M_dot_matrix_x1_1_block, 2 * M_dot_matrix_x1_1_block}, 1)}, 0) };

	Tensor M_dot_matrix_x1_2{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x1_2_block, M_dot_matrix_x1_2_block}, 1),
		torch::cat({M_dot_matrix_x1_2_block, 2 * M_dot_matrix_x1_2_block}, 1)}, 0) };

	Tensor M_dot_matrix_x1_3{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x1_3_block, M_dot_matrix_x1_3_block}, 1),
		torch::cat({M_dot_matrix_x1_3_block, 2 * M_dot_matrix_x1_3_block}, 1)}, 0) };

	Tensor M_dot_q_dot_x0_1{ torch::mm(((1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix_x0_1 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_x0_1), e.velocity_vec) };
	Tensor M_dot_q_dot_x0_2{ torch::mm(((1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix_x0_2 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_x0_2), e.velocity_vec) };
	Tensor M_dot_q_dot_x0_3{ torch::mm(((1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix_x0_3 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_x0_3), e.velocity_vec) };
	Tensor M_dot_q_dot_x1_1{ torch::mm(((1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix_x1_1 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_x1_1), e.velocity_vec) };
	Tensor M_dot_q_dot_x1_2{ torch::mm(((1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix_x1_2 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_x1_2), e.velocity_vec) };
	Tensor M_dot_q_dot_x1_3{ torch::mm(((1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix_x1_3 + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_x1_3), e.velocity_vec) };

	Tensor M_matrix_uv0_block{ torch::cat(
		{torch::cat({ZERO33, w_uv0}, 1),
		torch::cat({torch::transpose(w_uv0, 0, 1), wTw_uv0}, 1)}, 0) };

	Tensor M_matrix_uv0{ torch::cat(
		{torch::cat({2 * M_matrix_uv0_block, M_matrix_uv0_block}, 1),
		torch::cat({M_matrix_uv0_block, 2 * M_matrix_uv0_block}, 1)}, 0) };

	Tensor M_matrix_uv1_block{ torch::cat(
		{torch::cat({ZERO33, w_uv1}, 1),
		torch::cat({torch::transpose(w_uv1, 0, 1), wTw_uv1}, 1)}, 0) };

	Tensor M_matrix_uv1{ torch::cat(
		{torch::cat({2 * M_matrix_uv1_block, M_matrix_uv1_block}, 1),
		torch::cat({M_matrix_uv1_block, 2 * M_matrix_uv1_block}, 1)}, 0) };

	Tensor M_dot_matrix_uv0_block{
		torch::cat({ torch::cat({ZERO33, -w_dot_uv0}, 1),
			 torch::cat({torch::transpose(-w_dot_uv0, 0, 1), wTw_dot_uv0}, 1)}, 0) };

	Tensor M_dot_matrix_uv0{
		torch::cat({torch::cat({2 * M_dot_matrix_uv0_block, M_dot_matrix_uv0_block}, 1),
			torch::cat({M_dot_matrix_uv0_block, 2 * M_dot_matrix_uv0_block}, 1)}, 0) };

	Tensor M_dot_matrix_uv1_block{
		torch::cat({ torch::cat({ZERO33, -w_dot_uv1}, 1),
			 torch::cat({torch::transpose(-w_dot_uv1, 0, 1), wTw_dot_uv1}, 1)}, 0) };

	Tensor M_dot_matrix_uv1{
		torch::cat({torch::cat({2 * M_dot_matrix_uv1_block, M_dot_matrix_uv1_block}, 1),
			torch::cat({M_dot_matrix_uv1_block, 2 * M_dot_matrix_uv1_block}, 1)}, 0) };

	Tensor M_dot_q_dot_uv0{
		torch::mm(((1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix_uv0 -
		(1.0 / 6.0) * e.yarn.rho * M_dot + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_uv0), e.velocity_vec) };

	Tensor M_dot_q_dot_uv1{
		torch::mm(((1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix_uv1 -
		(1.0 / 6.0) * e.yarn.rho * M_dot + (1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_uv1), e.velocity_vec) };

	Tensor M_dot_q_dot_q{
		torch::cat({M_dot_q_dot_x0_1,
			M_dot_q_dot_x0_2,
			M_dot_q_dot_x0_3,
			M_dot_q_dot_uv0,
			M_dot_q_dot_x1_1,
			M_dot_q_dot_x1_2,
			M_dot_q_dot_x1_3,
			M_dot_q_dot_uv1,}, 1) };

	Tensor L0L0_MDotQDot{ M_dot_q_dot_q.index({Slice(0,3), Slice(0,3)}) };
	Tensor L0L1_MDotQDot{ M_dot_q_dot_q.index({Slice(0,3), Slice(4,7)}) };
	Tensor L1L0_MDotQDot{ M_dot_q_dot_q.index({Slice(4,7), Slice(0,3)}) };
	Tensor L1L1_MDotQDot{ M_dot_q_dot_q.index({Slice(4,7), Slice(4,7)}) };

	Tensor E0L0_MDotQDot{ M_dot_q_dot_q.index({3, Slice(0,3)}).view({1,-1}) };
	Tensor E0L1_MDotQDot{ M_dot_q_dot_q.index({3, Slice(4,7)}).view({1,-1}) };
	Tensor E1L0_MDotQDot{ M_dot_q_dot_q.index({7, Slice(0,3)}).view({1,-1}) };
	Tensor E1L1_MDotQDot{ M_dot_q_dot_q.index({7, Slice(4,7)}).view({1,-1}) };

	Tensor L0E0_MDotQDot{ M_dot_q_dot_q.index({Slice(0,3), 3}).view({-1,1}) };
	Tensor L0E1_MDotQDot{ M_dot_q_dot_q.index({Slice(0,3), 7}).view({-1,1}) };
	Tensor L1E0_MDotQDot{ M_dot_q_dot_q.index({Slice(4,7), 3}).view({-1,1}) };
	Tensor L1E1_MDotQDot{ M_dot_q_dot_q.index({Slice(4,7), 7}).view({-1,1}) };

	Tensor E0E0_MDotQDot{ M_dot_q_dot_q.index({3, 3}).view({1,1}) };
	Tensor E0E1_MDotQDot{ M_dot_q_dot_q.index({3, 7}).view({1,1}) };
	Tensor E1E0_MDotQDot{ M_dot_q_dot_q.index({7, 3}).view({1,1}) };
	Tensor E1E1_MDotQDot{ M_dot_q_dot_q.index({7, 7}).view({1,1}) };

	F_LL_Pos[L0L0_idx] = F_LL_Pos[L0L0_idx] + L0L0_MDotQDot;
	F_LL_Pos[L0L1_idx] = F_LL_Pos[L0L1_idx] + L0L1_MDotQDot;
	F_LL_Pos[L1L0_idx] = F_LL_Pos[L1L0_idx] + L1L0_MDotQDot;
	F_LL_Pos[L1L1_idx] = F_LL_Pos[L1L1_idx] + L1L1_MDotQDot;

	if (!(e.n0->is_edge)) {
		if (e.edge_type == "warp")
		{
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_EL_Pos[L0E0_idx] = F_EL_Pos[L0E0_idx] + E0L0_MDotQDot;
			F_LE_Pos[L0E0_idx] = F_LE_Pos[L0E0_idx] + L0E0_MDotQDot;
			F_EE_Pos[E0E0_idx] = F_EE_Pos[E0E0_idx] + E0E0_MDotQDot;
		}
		else
		{
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };

			F_EL_Pos[L0E0_idx] = F_EL_Pos[L0E0_idx] + E0L0_MDotQDot;
			F_LE_Pos[L0E0_idx] = F_LE_Pos[L0E0_idx] + L0E0_MDotQDot;
			F_EE_Pos[E0E0_idx] = F_EE_Pos[E0E0_idx] + E0E0_MDotQDot;
		}
	}

	if (!(e.n1->is_edge)) {
		if (e.edge_type == "warp")
		{
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };

			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_MDotQDot;
			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_MDotQDot;
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_MDotQDot;
		}
		else
		{
			int L1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1 };
			int E1E1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };

			F_EL_Pos[L1E1_idx] = F_EL_Pos[L1E1_idx] + E1L1_MDotQDot;
			F_LE_Pos[L1E1_idx] = F_LE_Pos[L1E1_idx] + L1E1_MDotQDot;
			F_EE_Pos[E1E1_idx] = F_EE_Pos[E1E1_idx] + E1E1_MDotQDot;
		}
	}

	if (!(e.n0->is_edge) && !(e.n1->is_edge)) {
		if (e.edge_type == "warp")
		{
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_LE_Pos[L1E0_idx] = F_LE_Pos[L1E0_idx] + L1E0_MDotQDot;
			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_MDotQDot;
			F_EL_Pos[L1E0_idx] = F_EL_Pos[L1E0_idx] + E0L1_MDotQDot;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_MDotQDot;
			F_EE_Pos[E0E1_idx] = F_EE_Pos[E0E1_idx] + E0E1_MDotQDot;
			F_EE_Pos[E1E0_idx] = F_EE_Pos[E1E0_idx] + E1E0_MDotQDot;
		}
		else {
			int L0E1_idx = find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1;
			int L1E0_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1;
			int E0E1_idx = find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv");
			int E1E0_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv");

			F_LE_Pos[L1E0_idx] = F_LE_Pos[L1E0_idx] + L1E0_MDotQDot;
			F_LE_Pos[L0E1_idx] = F_LE_Pos[L0E1_idx] + L0E1_MDotQDot;
			F_EL_Pos[L1E0_idx] = F_EL_Pos[L1E0_idx] + E0L1_MDotQDot;
			F_EL_Pos[L0E1_idx] = F_EL_Pos[L0E1_idx] + E1L0_MDotQDot;
			F_EE_Pos[E0E1_idx] = F_EE_Pos[E0E1_idx] + E0E1_MDotQDot;
			F_EE_Pos[E1E0_idx] = F_EE_Pos[E1E0_idx] + E1E0_MDotQDot;
		}
	}
}

void PhysicsCloth::MDotVDot_D_Vel(const Edge& e, Tensor& F_LL_Vel, Tensor& F_EL_Vel, Tensor& F_LE_Vel, Tensor& F_EE_Vel)
{
	int L0L0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L1L0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "LL", "NONE") },
		L0L1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") },
		L1L1_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "LL", "NONE") };

	Tensor delta_uv_dot;

	if (e.edge_type == "warp") {
		delta_uv_dot = e.n1->EVel[0] - e.n0->EVel[0];
	}
	else {
		delta_uv_dot = e.n1->EVel[1] - e.n0->EVel[1];
	}

	Tensor wTw{ torch::mm(torch::transpose(e.w, 0, 1), e.w) };
	Tensor w_dot{ ((e.n1->LVel - e.n0->LVel) * e.delta_uv - (e.n1->LPos - e.n0->LPos) * delta_uv_dot) / (e.delta_uv * e.delta_uv) };
	Tensor wTw_dot{ torch::mm(torch::transpose(e.w, 0, 1), w_dot) + torch::mm(torch::transpose(w_dot, 0, 1), e.w) };

	Tensor M_block{ torch::cat(
	{torch::cat({EYE3, -e.w}, 1),
	torch::cat({torch::transpose(-e.w,0,1), wTw}, 1)}, 0)
	};

	Tensor M_matrix{ torch::cat({ torch::cat({2 * M_block, M_block}, 1),
			torch::cat({M_block, 2 * M_block}, 1) }, 0) };

	Tensor M_dot_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot}, 1),
		torch::cat({torch::transpose(-w_dot, 0, 1), wTw_dot}, 1)}, 0) };

	Tensor M_dot{ (1.0 / 6.0) * e.yarn.rho * delta_uv_dot * M_matrix + (1.0 / 6.0) * e.yarn.rho * e.delta_uv *
		torch::cat({torch::cat({2 * M_dot_block, M_dot_block},1),
		torch::cat({M_dot_block, 2 * M_dot_block},1)}, 0) };

	Tensor w_dot_x0_dot{ (-1.0 / e.delta_uv) * EYE3 };
	Tensor w_dot_x1_dot{ (1.0 / e.delta_uv) * EYE3 };

	std::vector<Tensor> w_dot_x0_dot_xyz = w_dot_x0_dot.split(1, 1);
	std::vector<Tensor> w_dot_x1_dot_xyz = w_dot_x1_dot.split(1, 1);

	Tensor w_dot_x0_dot_1{ w_dot_x0_dot_xyz[0] },
		w_dot_x0_dot_2{ w_dot_x0_dot_xyz[1] },
		w_dot_x0_dot_3{ w_dot_x0_dot_xyz[2] };

	Tensor w_dot_x1_dot_1{ w_dot_x1_dot_xyz[0] },
		w_dot_x1_dot_2{ w_dot_x1_dot_xyz[1] },
		w_dot_x1_dot_3{ w_dot_x1_dot_xyz[2] };

	std::vector<Tensor> vel_der = torch::eye(8, opts).split(1, 1);

	Tensor vel_vector_x0_1{ vel_der[0] },
		vel_vector_x0_2{ vel_der[1] },
		vel_vector_x0_3{ vel_der[2] },
		vel_vector_uv0{ vel_der[3] },
		vel_vector_x1_1{ vel_der[4] },
		vel_vector_x1_2{ vel_der[5] },
		vel_vector_x1_3{ vel_der[6] },
		vel_vector_uv1{ vel_der[7] };

	Tensor wTw_dot_x0_dot_1{ 2 * torch::mm(torch::transpose(e.w,0,1), w_dot_x0_dot_1) };
	Tensor wTw_dot_x0_dot_2{ 2 * torch::mm(torch::transpose(e.w,0,1), w_dot_x0_dot_2) };
	Tensor wTw_dot_x0_dot_3{ 2 * torch::mm(torch::transpose(e.w,0,1), w_dot_x0_dot_3) };

	Tensor wTw_dot_x1_dot_1{ 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_x1_dot_1) };
	Tensor wTw_dot_x1_dot_2{ 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_x1_dot_2) };
	Tensor wTw_dot_x1_dot_3{ 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_x1_dot_3) };

	Tensor w_dot_uv0_dot{ (1.0 / e.delta_uv) * e.w };
	Tensor w_dot_uv1_dot{ (-1.0 / e.delta_uv) * e.w };

	Tensor wTw_dot_uv0_dot = 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_uv0_dot);
	Tensor wTw_dot_uv1_dot = 2 * torch::mm(torch::transpose(e.w, 0, 1), w_dot_uv1_dot);

	Tensor M_dot_matrix_x0_dot_1_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot_x0_dot_1}, 1),
		torch::cat({torch::transpose(-w_dot_x0_dot_1, 0 , 1), wTw_dot_x0_dot_1}, 1)}, 0)
	};

	Tensor M_dot_matrix_x0_dot_1{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x0_dot_1_block, M_dot_matrix_x0_dot_1_block}, 1),
		torch::cat({M_dot_matrix_x0_dot_1_block, 2 * M_dot_matrix_x0_dot_1_block}, 1)}, 0)
	};

	Tensor M_dot_matrix_x0_dot_2_block{ torch::cat(
	{torch::cat({ZERO33, -w_dot_x0_dot_2}, 1),
	torch::cat({torch::transpose(-w_dot_x0_dot_2, 0 , 1), wTw_dot_x0_dot_2}, 1)}, 0)
	};

	Tensor M_dot_matrix_x0_dot_2{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x0_dot_2_block, M_dot_matrix_x0_dot_2_block}, 1),
		torch::cat({M_dot_matrix_x0_dot_2_block, 2 * M_dot_matrix_x0_dot_2_block}, 1)}, 0)
	};

	Tensor M_dot_matrix_x0_dot_3_block{ torch::cat(
	{torch::cat({ZERO33, -w_dot_x0_dot_3}, 1),
	torch::cat({torch::transpose(-w_dot_x0_dot_3, 0 , 1), wTw_dot_x0_dot_3}, 1)}, 0)
	};

	Tensor M_dot_matrix_x0_dot_3{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x0_dot_3_block, M_dot_matrix_x0_dot_3_block}, 1),
		torch::cat({M_dot_matrix_x0_dot_3_block, 2 * M_dot_matrix_x0_dot_3_block}, 1)}, 0)
	};

	Tensor M_dot_matrix_x1_dot_1_block{ torch::cat(
	{torch::cat({ZERO33, -w_dot_x1_dot_1}, 1),
	torch::cat({torch::transpose(-w_dot_x1_dot_1, 0 , 1), wTw_dot_x1_dot_1}, 1)}, 0)
	};

	Tensor M_dot_matrix_x1_dot_1{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x1_dot_1_block, M_dot_matrix_x1_dot_1_block}, 1),
		torch::cat({M_dot_matrix_x1_dot_1_block, 2 * M_dot_matrix_x1_dot_1_block}, 1)}, 0)
	};

	Tensor M_dot_matrix_x1_dot_2_block{ torch::cat(
	{torch::cat({ZERO33, -w_dot_x1_dot_2}, 1),
	torch::cat({torch::transpose(-w_dot_x1_dot_2, 0 , 1), wTw_dot_x1_dot_2}, 1)}, 0)
	};

	Tensor M_dot_matrix_x1_dot_2{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x1_dot_2_block, M_dot_matrix_x1_dot_2_block}, 1),
		torch::cat({M_dot_matrix_x1_dot_2_block, 2 * M_dot_matrix_x1_dot_2_block}, 1)}, 0)
	};

	Tensor M_dot_matrix_x1_dot_3_block{ torch::cat(
	{torch::cat({ZERO33, -w_dot_x1_dot_3}, 1),
	torch::cat({torch::transpose(-w_dot_x1_dot_3, 0 , 1), wTw_dot_x1_dot_3}, 1)}, 0)
	};

	Tensor M_dot_matrix_x1_dot_3{ torch::cat(
		{torch::cat({2 * M_dot_matrix_x1_dot_3_block, M_dot_matrix_x1_dot_3_block}, 1),
		torch::cat({M_dot_matrix_x1_dot_3_block, 2 * M_dot_matrix_x1_dot_3_block}, 1)}, 0)
	};

	Tensor M_dot_q_dot_x0_dot_1{ (1.0 / 6.0) * e.yarn.rho * e.delta_uv * torch::mm(M_dot_matrix_x0_dot_1, e.velocity_vec) + torch::mm(M_dot, vel_vector_x0_1) };
	Tensor M_dot_q_dot_x0_dot_2{ (1.0 / 6.0) * e.yarn.rho * e.delta_uv * torch::mm(M_dot_matrix_x0_dot_2, e.velocity_vec) + torch::mm(M_dot, vel_vector_x0_2) };
	Tensor M_dot_q_dot_x0_dot_3{ (1.0 / 6.0) * e.yarn.rho * e.delta_uv * torch::mm(M_dot_matrix_x0_dot_3, e.velocity_vec) + torch::mm(M_dot, vel_vector_x0_3) };
	Tensor M_dot_q_dot_x1_dot_1{ (1.0 / 6.0) * e.yarn.rho * e.delta_uv * torch::mm(M_dot_matrix_x1_dot_1, e.velocity_vec) + torch::mm(M_dot, vel_vector_x1_1) };
	Tensor M_dot_q_dot_x1_dot_2{ (1.0 / 6.0) * e.yarn.rho * e.delta_uv * torch::mm(M_dot_matrix_x1_dot_2, e.velocity_vec) + torch::mm(M_dot, vel_vector_x1_2) };
	Tensor M_dot_q_dot_x1_dot_3{ (1.0 / 6.0) * e.yarn.rho * e.delta_uv * torch::mm(M_dot_matrix_x1_dot_3, e.velocity_vec) + torch::mm(M_dot, vel_vector_x1_3) };

	Tensor M_dot_matrix_uv0_dot_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot_uv0_dot}, 1),
		torch::cat({torch::transpose(-w_dot_uv0_dot, 0, 1), wTw_dot_uv0_dot}, 1),}, 0) };

	Tensor M_dot_matrix_uv0_dot{ torch::cat(
		{torch::cat({2 * M_dot_matrix_uv0_dot_block, M_dot_matrix_uv0_dot_block},1),
		torch::cat({M_dot_matrix_uv0_dot_block, 2 * M_dot_matrix_uv0_dot_block},1)}, 0) };

	Tensor M_dot_matrix_uv1_dot_block{ torch::cat(
		{torch::cat({ZERO33, -w_dot_uv1_dot}, 1),
		torch::cat({torch::transpose(-w_dot_uv1_dot, 0, 1), wTw_dot_uv1_dot}, 1),}, 0) };

	Tensor M_dot_matrix_uv1_dot{ torch::cat(
		{torch::cat({2 * M_dot_matrix_uv1_dot_block, M_dot_matrix_uv1_dot_block},1),
		torch::cat({M_dot_matrix_uv1_dot_block, 2 * M_dot_matrix_uv1_dot_block},1)}, 0) };

	Tensor M_dot_q_dot_uv0_dot{ torch::mm((-1.0 / 6.0) * e.yarn.rho * M_matrix +
		(1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_uv0_dot, e.velocity_vec) +
		torch::mm(M_dot, vel_vector_uv0) };

	Tensor M_dot_q_dot_uv1_dot{ torch::mm((1.0 / 6.0) * e.yarn.rho * M_matrix +
		(1.0 / 6.0) * e.yarn.rho * e.delta_uv * M_dot_matrix_uv1_dot, e.velocity_vec) +
		torch::mm(M_dot, vel_vector_uv1) };

	Tensor M_dot_q_dot_q_dot{ torch::cat(
		{M_dot_q_dot_x0_dot_1,
		M_dot_q_dot_x0_dot_2,
		M_dot_q_dot_x0_dot_3,
		M_dot_q_dot_uv0_dot,
		M_dot_q_dot_x1_dot_1,
		M_dot_q_dot_x1_dot_2,
		M_dot_q_dot_x1_dot_3,
		M_dot_q_dot_uv1_dot}, 1) };

	Tensor L0L0_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({Slice(0,3), Slice(0,3)}) };
	Tensor L0L1_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({Slice(0,3), Slice(4,7)}) };
	Tensor L1L0_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({Slice(4,7), Slice(0,3)}) };
	Tensor L1L1_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({Slice(4,7), Slice(4,7)}) };

	Tensor E0L0_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({3, Slice(0,3)}).view({1,-1}) };
	Tensor E0L1_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({3, Slice(4,7)}).view({1,-1}) };
	Tensor E1L0_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({7, Slice(0,3)}).view({1,-1}) };
	Tensor E1L1_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({7, Slice(4,7)}).view({1,-1}) };

	Tensor L0E0_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({Slice(0,3), 3}).view({-1,1}) };
	Tensor L0E1_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({Slice(0,3), 7}).view({-1,1}) };
	Tensor L1E0_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({Slice(4,7), 3}).view({-1,1}) };
	Tensor L1E1_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({Slice(4,7), 7}).view({-1,1}) };

	Tensor E0E0_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({3, 3}).view({1,1}) };
	Tensor E0E1_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({3, 7}).view({1,1}) };
	Tensor E1E0_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({7, 3}).view({1,1}) };
	Tensor E1E1_M_dot_q_dot_q_dot{ M_dot_q_dot_q_dot.index({7, 7}).view({1,1}) };

	F_LL_Vel[L0L0_idx] = F_LL_Vel[L0L0_idx] - L0L0_M_dot_q_dot_q_dot;
	F_LL_Vel[L0L1_idx] = F_LL_Vel[L0L1_idx] - L0L1_M_dot_q_dot_q_dot;
	F_LL_Vel[L1L0_idx] = F_LL_Vel[L1L0_idx] - L1L0_M_dot_q_dot_q_dot;
	F_LL_Vel[L1L1_idx] = F_LL_Vel[L1L1_idx] - L1L1_M_dot_q_dot_q_dot;

	if (!(e.n0->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E0_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_EL_Vel[L0E0_idx] = F_EL_Vel[L0E0_idx] - E0L0_M_dot_q_dot_q_dot;
			F_LE_Vel[L0E0_idx] = F_LE_Vel[L0E0_idx] - L0E0_M_dot_q_dot_q_dot;
			F_EE_Vel[E0E0_idx] = F_EE_Vel[E0E0_idx] - E0E0_M_dot_q_dot_q_dot;
		}
		else {
			int L0E0_idx = find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1;
			int E0E0_idx = find_index(e.n0->idx_u, e.n0->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv");

			F_EL_Vel[L0E0_idx] = F_EL_Vel[L0E0_idx] - E0L0_M_dot_q_dot_q_dot;
			F_LE_Vel[L0E0_idx] = F_LE_Vel[L0E0_idx] - L0E0_M_dot_q_dot_q_dot;
			F_EE_Vel[E0E0_idx] = F_EE_Vel[E0E0_idx] - E0E0_M_dot_q_dot_q_dot;
		}
	}

	if (!(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L1E1_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE");
			int E1E1_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu");

			F_EL_Vel[L1E1_idx] = F_EL_Vel[L1E1_idx] - E1L1_M_dot_q_dot_q_dot;
			F_LE_Vel[L1E1_idx] = F_LE_Vel[L1E1_idx] - L1E1_M_dot_q_dot_q_dot;
			F_EE_Vel[E1E1_idx] = F_EE_Vel[E1E1_idx] - E1E1_M_dot_q_dot_q_dot;
		}
		else {
			int L1E1_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1;
			int E1E1_idx = find_index(e.n1->idx_u, e.n1->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv");

			F_EL_Vel[L1E1_idx] = F_EL_Vel[L1E1_idx] - E1L1_M_dot_q_dot_q_dot;
			F_LE_Vel[L1E1_idx] = F_LE_Vel[L1E1_idx] - L1E1_M_dot_q_dot_q_dot;
			F_EE_Vel[E1E1_idx] = F_EE_Vel[E1E1_idx] - E1E1_M_dot_q_dot_q_dot;
		}
	}

	if (!(e.n0->is_edge) && !(e.n1->is_edge)) {
		if (e.edge_type == "warp") {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "uu") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "uu") };

			F_LE_Vel[L1E0_idx] = F_LE_Vel[L1E0_idx] - L1E0_M_dot_q_dot_q_dot;
			F_LE_Vel[L0E1_idx] = F_LE_Vel[L0E1_idx] - L0E1_M_dot_q_dot_q_dot;
			F_EL_Vel[L1E0_idx] = F_EL_Vel[L1E0_idx] - E0L1_M_dot_q_dot_q_dot;
			F_EL_Vel[L0E1_idx] = F_EL_Vel[L0E1_idx] - E1L0_M_dot_q_dot_q_dot;
			F_EE_Vel[E0E1_idx] = F_EE_Vel[E0E1_idx] - E0E1_M_dot_q_dot_q_dot;
			F_EE_Vel[E1E0_idx] = F_EE_Vel[E1E0_idx] - E1E0_M_dot_q_dot_q_dot;
		}
		else {
			int L0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EL", "NONE") + 1 };
			int L1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EL", "NONE") + 1 };
			int E0E1_idx{ find_index(e.n0->idx_u, e.n0->idx_v, e.n1->idx_u, e.n1->idx_v, "EE", "vv") };
			int E1E0_idx{ find_index(e.n1->idx_u, e.n1->idx_v, e.n0->idx_u, e.n0->idx_v, "EE", "vv") };

			F_LE_Vel[L1E0_idx] = F_LE_Vel[L1E0_idx] - L1E0_M_dot_q_dot_q_dot;
			F_LE_Vel[L0E1_idx] = F_LE_Vel[L0E1_idx] - L0E1_M_dot_q_dot_q_dot;
			F_EL_Vel[L1E0_idx] = F_EL_Vel[L1E0_idx] - E0L1_M_dot_q_dot_q_dot;
			F_EL_Vel[L0E1_idx] = F_EL_Vel[L0E1_idx] - E1L0_M_dot_q_dot_q_dot;
			F_EE_Vel[E0E1_idx] = F_EE_Vel[E0E1_idx] - E0E1_M_dot_q_dot_q_dot;
			F_EE_Vel[E1E0_idx] = F_EE_Vel[E1E0_idx] - E1E0_M_dot_q_dot_q_dot;
		}
	}
}

void PhysicsCloth::ConSlideFriction(Node& n, Tensor& F_E_is_Slide,
	Tensor& F_E_No_Slide, Tensor& F_EE_Pos_No_Slide, Tensor& F_EE_Vel_No_Slide,
	Tensor& F_E, Tensor& F_EE_Pos, Tensor& F_EE_Vel,
	Tensor& StrhBend_C_L, Tensor& StrhBend_NC_L,
	Tensor& StrhBend_C_LE, Tensor& StrhBend_NC_LE)
{
	int FL_idx{ n.idx_u + (n.idx_v * env->cloth->width) };
	int FE_idx{ (n.idx_u - 1 + (env->cloth->width - 2) * (n.idx_v - 1)) };

	int LE_idx_u{ find_index(n.idx_u, n.idx_v, n.idx_u, n.idx_v, "EL", "NONE") };
	int LE_idx_v{ LE_idx_u + 1 };
	int EE_idx_u{ find_index(n.idx_u, n.idx_v, n.idx_u, n.idx_v, "EE", "uu") };
	int EE_idx_v{ find_index(n.idx_u, n.idx_v, n.idx_u, n.idx_v, "EE", "vv") };

	Tensor uv_minus_uvbar{ n.EPos - n.EPosBar };
	Tensor F_EE_Pos_u{ F_EE_Pos_No_Slide[EE_idx_u] };
	Tensor F_EE_Pos_v{ F_EE_Pos_No_Slide[EE_idx_v] };
	Tensor F_EE_Vel_u{ F_EE_Vel_No_Slide[EE_idx_u] };
	Tensor F_EE_Vel_v{ F_EE_Vel_No_Slide[EE_idx_v] };

	Tensor compression, compression_EE_u, compression_EE_v;

	compression = torch::relu(0.5 * torch::mm(n.n.transpose(1, 0), StrhBend_C_L[FL_idx]));
	compression_EE_u = 0.5 * torch::mm(n.n.transpose(1, 0), StrhBend_C_LE[LE_idx_u]);
	compression_EE_v = 0.5 * torch::mm(n.n.transpose(1, 0), StrhBend_C_LE[LE_idx_v]);
	n.comp = compression;

	//if (floor(torch::norm(StrhBend_NC_L[FL_idx]).item<double>() * 1e5) == 0) {
	//	compression = torch::relu(0.5 * torch::mm(n.n.transpose(1, 0), StrhBend_C_L[FL_idx]));
	//	compression_EE_u = 0.5 * torch::mm(n.n.transpose(1, 0), StrhBend_C_LE[LE_idx_u]);
	//	compression_EE_v = 0.5 * torch::mm(n.n.transpose(1, 0), StrhBend_C_LE[LE_idx_v]);
	//	n.comp = compression;
	//}
	//else {
	//	compression = torch::relu(0.5 * torch::mm(n.n.transpose(1, 0), StrhBend_NC_L[FL_idx]));
	//	compression_EE_u = 0.5 * torch::mm(n.n.transpose(1, 0), StrhBend_NC_LE[LE_idx_u]);
	//	compression_EE_v = 0.5 * torch::mm(n.n.transpose(1, 0), StrhBend_NC_LE[LE_idx_v]);
	//	n.comp = compression;
	//}

	Tensor fric_lim{ torch::ones({2,1}, opts) * compression * env->cloth->mu };
	Tensor is_slide{ torch::sigmoid(1e4 * (torch::abs(F_E_No_Slide[FE_idx]) - fric_lim)) }; // Soft Conditional Function
	//Tensor is_slide{ torch::gt(torch::abs( F_E_No_Slide[FE_idx]), fric_lim) }; // Discontinuous Conditional Function
	//Tensor is_stick{ torch::logical_not(is_slide) };

	Tensor fric_force{ (-0.5 * (env->cloth->kf * uv_minus_uvbar - torch::tanh(10 * uv_minus_uvbar) * compression * env->cloth->mu) * torch::tanh(10 * (fric_lim - F_E[FE_idx])) + 0.5 * (env->cloth->kf * uv_minus_uvbar + torch::tanh(10 * uv_minus_uvbar) * compression * env->cloth->mu)) - env->cloth->df * n.EVel };

	Tensor fric_u_pos{ (-0.5 * (env->cloth->kf - (torch::ones({1,1}, opts) - torch::pow(torch::tanh(10 * uv_minus_uvbar[0]),2)) * env->cloth->mu * compression - torch::tanh(10 * uv_minus_uvbar[0]) * env->cloth->mu * compression_EE_u)) * torch::tanh(10 * (fric_lim - F_E_No_Slide[FE_idx])[0]) - (0.5 * (env->cloth->kf * uv_minus_uvbar[0] - torch::tanh(10 * uv_minus_uvbar[0]) * env->cloth->mu * compression)) * (torch::ones({1,1}, opts) - torch::pow(torch::tanh(10 * (fric_lim - F_E_No_Slide[FE_idx])[0]),2)) * (F_EE_Pos_u - env->cloth->mu * compression_EE_u) - (0.5 * (env->cloth->kf + (torch::ones({1,1}, opts) - torch::pow(torch::tanh(10 * uv_minus_uvbar[0]),2)) * env->cloth->mu * compression + torch::tanh(10 * uv_minus_uvbar[0]) * env->cloth->mu * compression_EE_u)) };

	Tensor fric_v_pos{ (-0.5 * (env->cloth->kf - (torch::ones({1,1}, opts) - torch::pow(torch::tanh(10 * uv_minus_uvbar[1]),2)) * env->cloth->mu * compression - torch::tanh(10 * uv_minus_uvbar[1]) * env->cloth->mu * compression_EE_v)) * torch::tanh(10 * (fric_lim - F_E_No_Slide[FE_idx])[1]) - (0.5 * (env->cloth->kf * uv_minus_uvbar[1] - torch::tanh(10 * uv_minus_uvbar[1]) * env->cloth->mu * compression)) * (torch::ones({1,1}, opts) - torch::pow(torch::tanh(10 * (fric_lim - F_E_No_Slide[FE_idx])[1]),2)) * (F_EE_Pos_v - env->cloth->mu * compression_EE_v) - (0.5 * (env->cloth->kf + (torch::ones({1,1}, opts) - torch::pow(torch::tanh(10 * uv_minus_uvbar[1]),2)) * env->cloth->mu * compression + torch::tanh(10 * uv_minus_uvbar[1]) * env->cloth->mu * compression_EE_v)) };

	Tensor fric_u_vel{ (-0.5 * (env->cloth->kf * uv_minus_uvbar[0] - torch::tanh(10 * uv_minus_uvbar[0]) * env->cloth->mu * compression)) * (torch::ones({1}, opts) - torch::pow(torch::tanh((10 * fric_lim - F_E_No_Slide[FE_idx])[0]),2)) * F_EE_Vel_u - env->cloth->df.view({1,1}) };
	Tensor fric_v_vel{ (-0.5 * (env->cloth->kf * uv_minus_uvbar[1] - torch::tanh(10 * uv_minus_uvbar[1]) * env->cloth->mu * compression)) * (torch::ones({1}, opts) - torch::pow(torch::tanh((10 * fric_lim - F_E_No_Slide[FE_idx])[1]),2)) * F_EE_Vel_v - env->cloth->df.view({1,1}) };

	F_E_is_Slide[FE_idx] = is_slide;
	F_E[FE_idx] = F_E[FE_idx] + fric_force;

	F_EE_Pos[EE_idx_u] = F_EE_Pos[EE_idx_u] + fric_u_pos;
	F_EE_Pos[EE_idx_v] = F_EE_Pos[EE_idx_v] + fric_v_pos;

	F_EE_Vel[EE_idx_u] = F_EE_Vel[EE_idx_u] + fric_u_vel;
	F_EE_Vel[EE_idx_v] = F_EE_Vel[EE_idx_v] + fric_v_vel;
}

void PhysicsCloth::Shear_F(const Shear_Seg& s, Tensor& F_L)
{
	int L0_idx{ s.n0->idx_u + s.n0->idx_v * env->cloth->width },
		L1_idx{ s.n1->idx_u + s.n1->idx_v * env->cloth->width },
		L3_idx{ s.n3->idx_u + s.n3->idx_v * env->cloth->width };

	int c = 3; // Deciding shearing parameter increasing rate whilst "shear lock" 
	static double sig = 0.6; // Parameter decides transition smoothness

	Tensor comp_shear = s.n0->comp; // Compression at crossing node

	Tensor kx = (comp_shear + 1.0) * env->cloth->S * env->cloth->R * env->cloth->R * PI; // Non-jamming shear stiffness

	if (s.is_jamming) {
		auto phi_bar{ torch::tensor({ PI / 2 }, opts) };
		auto phi_l{ torch::tensor({ env->cloth->jamming_thrd }, opts) };

		auto phi_x1{ -torch::mm(s.P1, s.d3) / (s.l1 * torch::sin(s.phi)) };
		auto phi_x3{ -torch::mm(s.P3, s.d1) / (s.l3 * torch::sin(s.phi)) };

		auto gamma = (env->cloth->L * sqrt(2.0) - 2 * env->cloth->L * torch::sin(0.5 * s.phi)) / env->cloth->R;
		auto gamma_x1 = -(env->cloth->L / env->cloth->R) * torch::cos(0.5 * s.phi) * phi_x1;
		auto gamma_x3 = -(env->cloth->L / env->cloth->R) * torch::cos(0.5 * s.phi) * phi_x3;

		auto phi_s = s.phi * (s.phi - phi_l) * (s.phi - phi_bar);
		auto phi_s_x1 = phi_x1 * (s.phi - phi_l) * (s.phi - phi_bar)
			+ s.phi * phi_x1 * (s.phi - phi_bar)
			+ s.phi * (s.phi - phi_l) * phi_x1;
		auto phi_s_x3 = phi_x3 * (s.phi - phi_l) * (s.phi - phi_bar)
			+ s.phi * phi_x3 * (s.phi - phi_bar)
			+ s.phi * (s.phi - phi_l) * phi_x3;

		auto con_func_x_num = torch::pow(phi_bar, 5) * (s.phi - phi_l);
		auto con_func_x_don = (torch::pow(phi_s, 2) + torch::pow(phi_bar, 4) * sig * sig);
		auto con_func_x = con_func_x_num / con_func_x_don;
		auto con_func = torch::tanh(con_func_x);

		auto con_func_x_num_x1 = torch::pow(phi_bar, 5) * phi_x1;
		auto con_func_x_don_x1 = 2 * phi_s * phi_s_x1;

		auto con_func_x_num_x3 = torch::pow(phi_bar, 5) * phi_x3;
		auto con_func_x_don_x3 = 2 * phi_s * phi_s_x3;

		auto con_func_x_x1_num = con_func_x_num_x1 * con_func_x_don - con_func_x_num * con_func_x_don_x1;
		auto con_func_x_x1_don = torch::pow(con_func_x_don, 2);
		auto con_func_x_x1 = con_func_x_x1_num / con_func_x_x1_don;

		auto con_func_x_x3_num = con_func_x_num_x3 * con_func_x_don - con_func_x_num * con_func_x_don_x3;
		auto con_func_x_x3_don = torch::pow(con_func_x_don, 2);
		auto con_func_x_x3 = con_func_x_x3_num / con_func_x_x3_don;

		kx = 0.5 * (comp_shear + 1.0) * env->cloth->S * PI * pow(env->cloth->R, 2)
			* ((1 + torch::pow(gamma, c)) + (1 - torch::pow(gamma, c)) * con_func);

		auto kx_x1 = 0.5 * (comp_shear + 1.0) * env->cloth->S * PI * pow(env->cloth->R, 2) *
			(c * torch::pow(gamma, c - 1) * gamma_x1 - c * torch::pow(gamma, c - 1) * gamma_x1 * con_func
				+ (1 - torch::pow(gamma, c)) * (1 - torch::pow(con_func, 2) * con_func_x_x1));

		auto kx_x3 = 0.5 * (comp_shear + 1.0) * env->cloth->S * PI * pow(env->cloth->R, 2) *
			(c * torch::pow(gamma, c - 1) * gamma_x3 - c * torch::pow(gamma, c - 1) * gamma_x3 * con_func
				+ (1 - torch::pow(gamma, c)) * (1 - torch::pow(con_func, 2) * con_func_x_x3));

		auto F_x1 = -0.5 * kx_x1 * env->cloth->L * torch::pow(s.phi - phi_bar, 2)
			+ (kx * env->cloth->L * (s.phi - phi_bar)) / (s.l1 * torch::sin(s.phi)) * torch::mm(s.P1, s.d3);

		auto F_x3 = -0.5 * kx_x3 * env->cloth->L * torch::pow(s.phi - phi_bar, 2)
			+ (kx * env->cloth->L * (s.phi - phi_bar)) / (s.l3 * torch::sin(s.phi)) * torch::mm(s.P3, s.d1);

		auto F_x0 = -(F_x1 + F_x3);

		F_L[L0_idx] = F_L[L0_idx] + F_x0;
		F_L[L1_idx] = F_L[L1_idx] + F_x1;
		F_L[L3_idx] = F_L[L3_idx] + F_x3;

		return;
	}

	auto F_x1 = (kx * env->cloth->L * (s.phi.item<double>() - PI / 2)) / (s.l1 * torch::sin(s.phi)) * torch::mm(s.P1, s.d3);
	auto F_x3 = (kx * env->cloth->L * (s.phi.item<double>() - PI / 2)) / (s.l3 * torch::sin(s.phi)) * torch::mm(s.P3, s.d1);
	auto F_x0 = -1 * (F_x1 + F_x3);

	F_L[L0_idx] = F_L[L0_idx] + F_x0;
	F_L[L1_idx] = F_L[L1_idx] + F_x1;
	F_L[L3_idx] = F_L[L3_idx] + F_x3;

}

void PhysicsCloth::Shear_D(const Shear_Seg& s, Tensor& F_LL_Pos)
{
	int L0L0_idx{ find_index(s.n0->idx_u, s.n0->idx_v, s.n0->idx_u, s.n0->idx_v, "LL", "NONE") },
		L0L1_idx{ find_index(s.n0->idx_u, s.n0->idx_v, s.n1->idx_u, s.n1->idx_v, "LL", "NONE") },
		L0L3_idx{ find_index(s.n0->idx_u, s.n0->idx_v, s.n3->idx_u, s.n3->idx_v, "LL", "NONE") },
		L1L0_idx{ find_index(s.n1->idx_u, s.n1->idx_v, s.n0->idx_u, s.n0->idx_v, "LL", "NONE") },
		L1L1_idx{ find_index(s.n1->idx_u, s.n1->idx_v, s.n1->idx_u, s.n1->idx_v, "LL", "NONE") },
		L1L3_idx{ find_index(s.n1->idx_u, s.n1->idx_v, s.n3->idx_u, s.n3->idx_v, "LL", "NONE") },
		L3L0_idx{ find_index(s.n3->idx_u, s.n3->idx_v, s.n0->idx_u, s.n0->idx_v, "LL", "NONE") },
		L3L1_idx{ find_index(s.n3->idx_u, s.n3->idx_v, s.n1->idx_u, s.n1->idx_v, "LL", "NONE") },
		L3L3_idx{ find_index(s.n3->idx_u, s.n3->idx_v, s.n3->idx_u, s.n3->idx_v, "LL", "NONE") };

	int c = 3; // Deciding shearing parameter increasing rate whilst "shear lock" 
	static double sig = 0.6; // Parameter decides transition smoothness

	Tensor comp_shear = s.n0->comp; // Compression at crossing node

	Tensor kx = (comp_shear + 1.0) * env->cloth->S * env->cloth->R * env->cloth->R * PI; // Non-jamming shear stiffness

	if (s.is_jamming) {
		Tensor phi_bar = torch::tensor({ PI / 2 }, opts);
		Tensor phi_l = torch::tensor({ env->cloth->jamming_thrd }, opts);

		auto phi_x1{ -torch::mm(s.P1, s.d3) / (s.l1 * torch::sin(s.phi)) };
		auto phi_x3{ -torch::mm(s.P3, s.d1) / (s.l3 * torch::sin(s.phi)) };

		auto phi_x1_x1 = (torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d1.squeeze())
			+ torch::mm(torch::outer(s.d1.squeeze(), s.d3.squeeze()), s.P1)
			- (torch::cos(s.phi) / torch::pow(torch::sin(s.phi), 2)
				* torch::mm(torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d3.squeeze()), s.P1))
			+ torch::cos(s.phi) * s.P1) / (torch::pow(s.l1, 2) * torch::sin(s.phi));

		auto phi_x3_x3 = (torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d3.squeeze())
			+ torch::mm(torch::outer(s.d3.squeeze(), s.d1.squeeze()), s.P3)
			- (torch::cos(s.phi) / torch::pow(torch::sin(s.phi), 2)
				* torch::mm(torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d1.squeeze()), s.P3))
			+ torch::cos(s.phi) * s.P3) / (torch::pow(s.l3, 2) * torch::sin(s.phi));

		auto phi_x1_x3 = -((torch::cos(s.phi) / torch::pow(torch::sin(s.phi), 2))
			* torch::mm(torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d1.squeeze()), s.P3)
			+ torch::mm(s.P1, s.P3)) / (s.l1 * s.l3 * torch::sin(s.phi));

		auto phi_x3_x1 = -((torch::cos(s.phi) / torch::pow(torch::sin(s.phi), 2))
			* torch::mm(torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d3.squeeze()), s.P3)
			+ torch::mm(s.P3, s.P1)) / (s.l3 * s.l1 * torch::sin(s.phi));

		auto gamma = (env->cloth->L * sqrt(2.0) - 2 * env->cloth->L * torch::sin(0.5 * s.phi)) / env->cloth->R;
		auto gamma_x1 = -(env->cloth->L / env->cloth->R) * torch::cos(0.5 * s.phi) * phi_x1;
		auto gamma_x3 = -(env->cloth->L / env->cloth->R) * torch::cos(0.5 * s.phi) * phi_x3;

		auto gamma_x1_x1 = (0.5 * (env->cloth->L / env->cloth->R) * torch::sin(0.5 * s.phi)
			* torch::mm(torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d3.squeeze()), s.P1))
			/ torch::pow(s.l1 * torch::sin(s.phi), 2)
			- (env->cloth->L / env->cloth->R) * torch::cos(0.5 * s.phi) * phi_x1_x1;

		auto gamma_x3_x3 = (0.5 * (env->cloth->L / env->cloth->R) * torch::sin(0.5 * s.phi)
			* torch::mm(torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d1.squeeze()), s.P3))
			/ torch::pow(s.l3 * torch::sin(s.phi), 2)
			- (env->cloth->L / env->cloth->R) * torch::cos(0.5 * s.phi) * phi_x3_x3;

		auto gamma_x1_x3 = (0.5 * (env->cloth->L / env->cloth->R) * torch::sin(s.phi / 2) *
			torch::mm(torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d1.squeeze()), s.P3))
			/ (s.l1 * s.l3 * torch::pow(torch::sin(s.phi), 2))
			- (env->cloth->L / env->cloth->R) * torch::cos(s.phi / 2) * phi_x1_x3;

		auto gamma_x3_x1 = (0.5 * (env->cloth->L / env->cloth->R) * torch::sin(s.phi / 2) *
			torch::mm(torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d3.squeeze()), s.P1))
			/ (s.l3 * s.l1 * torch::pow(torch::sin(s.phi), 2))
			- (env->cloth->L / env->cloth->R) * torch::cos(s.phi / 2) * phi_x3_x1;

		auto phi_s = s.phi * (s.phi - phi_l) * (s.phi - phi_bar);

		auto phi_s_x1 = phi_x1 * (s.phi - phi_l) * (s.phi - phi_bar)
			+ s.phi * phi_x1 * (s.phi - phi_bar)
			+ s.phi * (s.phi - phi_l) * phi_x1;

		auto phi_s_x3 = phi_x3 * (s.phi - phi_l) * (s.phi - phi_bar)
			+ s.phi * phi_x3 * (s.phi - phi_bar)
			+ s.phi * (s.phi - phi_l) * phi_x3;

		auto phi_s_x1_x1 = phi_x1_x1 * (s.phi - phi_l) * (s.phi - phi_bar)
			+ torch::outer(phi_x1.squeeze(), phi_x1.squeeze()) * (s.phi - phi_bar)
			+ torch::outer(phi_x1.squeeze(), phi_x1.squeeze()) * (s.phi - phi_l)
			+ torch::outer(phi_x1.squeeze(), phi_x1.squeeze()) * (s.phi - phi_bar)
			+ s.phi * phi_x1_x1 * (s.phi - phi_bar)
			+ s.phi * torch::outer(phi_x1.squeeze(), phi_x1.squeeze())
			+ torch::outer(phi_x1.squeeze(), phi_x1.squeeze()) * (s.phi - phi_l)
			+ s.phi * torch::outer(phi_x1.squeeze(), phi_x1.squeeze())
			+ s.phi * (s.phi - phi_l) * phi_x1_x1;

		auto phi_s_x3_x3 = phi_x3_x3 * (s.phi - phi_l) * (s.phi - phi_bar)
			+ torch::outer(phi_x3.squeeze(), phi_x3.squeeze()) * (s.phi - phi_bar)
			+ torch::outer(phi_x3.squeeze(), phi_x3.squeeze()) * (s.phi - phi_l)
			+ torch::outer(phi_x3.squeeze(), phi_x3.squeeze()) * (s.phi - phi_bar)
			+ s.phi * phi_x3_x3 * (s.phi - phi_bar)
			+ s.phi * torch::outer(phi_x3.squeeze(), phi_x3.squeeze())
			+ torch::outer(phi_x3.squeeze(), phi_x3.squeeze()) * (s.phi - phi_l)
			+ s.phi * torch::outer(phi_x3.squeeze(), phi_x3.squeeze())
			+ s.phi * (s.phi - phi_l) * phi_x3_x3;

		auto phi_s_x1_x3 = (phi_x1_x3 * (s.phi - phi_l) * (s.phi - phi_bar)
			+ torch::outer(phi_x1.squeeze(), phi_x3.squeeze()) * (s.phi - phi_bar)
			+ torch::outer(phi_x1.squeeze(), phi_x3.squeeze()) * (s.phi - phi_l)
			+ torch::outer(phi_x3.squeeze(), phi_x1.squeeze()) * (s.phi - phi_bar)
			+ s.phi * phi_x1_x3 * (s.phi - phi_bar)
			+ s.phi * torch::outer(phi_x1.squeeze(), phi_x3.squeeze())
			+ torch::outer(phi_x3.squeeze(), phi_x1.squeeze()) * (s.phi - phi_l)
			+ s.phi * torch::outer(phi_x3.squeeze(), phi_x1.squeeze())
			+ s.phi * (s.phi - phi_l) * phi_x1_x3);

		auto phi_s_x3_x1 = (phi_x3_x1 * (s.phi - phi_l) * (s.phi - phi_bar)
			+ torch::outer(phi_x3.squeeze(), phi_x1.squeeze()) * (s.phi - phi_bar)
			+ torch::outer(phi_x3.squeeze(), phi_x1.squeeze()) * (s.phi - phi_l)
			+ torch::outer(phi_x3.squeeze(), phi_x1.squeeze()) * (s.phi - phi_bar)
			+ s.phi * phi_x3_x1 * (s.phi - phi_bar)
			+ s.phi * torch::outer(phi_x3.squeeze(), phi_x1.squeeze())
			+ torch::outer(phi_x3.squeeze(), phi_x1.squeeze()) * (s.phi - phi_l)
			+ s.phi * torch::outer(phi_x3.squeeze(), phi_x1.squeeze())
			+ s.phi * (s.phi - phi_l) * phi_x3_x1);

		auto con_func_x_num = torch::pow(phi_bar, 5) * (s.phi - phi_l);
		auto con_func_x_den = (torch::pow(phi_s, 2) + torch::pow(phi_bar, 4) * sig * sig);
		auto con_func_x = con_func_x_num / con_func_x_den;
		auto con_func = torch::tanh(con_func_x);

		auto con_func_x_num_x1 = torch::pow(phi_bar, 5) * phi_x1;
		auto con_func_x_den_x1 = 2 * phi_s * phi_s_x1;

		auto con_func_x_num_x3 = torch::pow(phi_bar, 5) * phi_x3;
		auto con_func_x_den_x3 = 2 * phi_s * phi_s_x3;

		auto con_func_x_x1_num = con_func_x_num_x1 * con_func_x_den - con_func_x_num * con_func_x_den_x1;
		auto con_func_x_x1_den = torch::pow(con_func_x_den, 2);
		auto con_func_x_x1 = con_func_x_x1_num / con_func_x_x1_den;
		auto con_func_x1 = (1.0 - torch::pow(con_func, 2)) * con_func_x_x1;

		auto con_func_x_x3_num = con_func_x_num_x3 * con_func_x_den - con_func_x_num * con_func_x_den_x3;
		auto con_func_x_x3_den = torch::pow(con_func_x_den, 2);
		auto con_func_x_x3 = con_func_x_x3_num / con_func_x_x3_den;
		auto con_func_x3 = (1.0 - torch::pow(con_func, 2)) * con_func_x_x3;

		auto con_func_x_num_x1_x1 = torch::pow(phi_bar, 5) * phi_x1_x1;
		auto con_func_x_num_x3_x3 = torch::pow(phi_bar, 5) * phi_x3_x3;
		auto con_func_x_num_x1_x3 = torch::pow(phi_bar, 5) * phi_x1_x3;
		auto con_func_x_num_x3_x1 = torch::pow(phi_bar, 5) * phi_x3_x1;

		auto con_func_x_x1_num_x1 = con_func_x_num_x1_x1 * con_func_x_den
			- 2 * con_func_x_num * torch::outer(phi_s_x1.squeeze(), phi_s_x1.squeeze())
			- 2 * con_func_x_num * phi_s * phi_s_x1_x1;
		auto con_func_x_x1_den_x1 = 2 * con_func_x_den * con_func_x_den_x1;
		auto con_func_x_x1_x1 = (con_func_x_x1_num_x1 * con_func_x_x1_den
			- torch::outer(con_func_x_x1_num.squeeze(), con_func_x_x1_den_x1.squeeze()))
			/ torch::pow(con_func_x_x1_den, 2);

		auto con_func_x_x3_num_x3 = con_func_x_num_x3_x3 * con_func_x_den
			- 2 * con_func_x_num * torch::outer(phi_s_x3.squeeze(), phi_s_x3.squeeze())
			- 2 * con_func_x_num * phi_s * phi_s_x3_x3;
		auto con_func_x_x3_den_x3 = 2 * con_func_x_den * con_func_x_den_x3;
		auto con_func_x_x3_x3 = (con_func_x_x3_num_x3 * con_func_x_x3_den
			- torch::outer(con_func_x_x3_num.squeeze(), con_func_x_x3_den_x3.squeeze()))
			/ torch::pow(con_func_x_x3_den, 2);

		auto con_func_x_x1_num_x3 = (con_func_x_num_x1_x3 * con_func_x_den
			+ 2 * phi_s * torch::outer(con_func_x_num_x1.squeeze(), phi_s_x3.squeeze())
			- 2 * phi_s * torch::outer(con_func_x_num_x3.squeeze(), phi_s_x1.squeeze())
			- 2 * con_func_x_num * torch::outer(phi_s_x1.squeeze(), phi_s_x3.squeeze())
			- 2 * con_func_x_num * phi_s * phi_s_x1_x3);
		auto con_func_x_x1_den_x3 = 2 * con_func_x_den * con_func_x_den_x3;
		auto con_func_x_x1_x3 = (con_func_x_x1_num_x3 * con_func_x_x1_den
			- torch::outer(con_func_x_x1_num.squeeze(), con_func_x_x1_den_x3.squeeze()))
			/ torch::pow(con_func_x_x1_den, 2);

		auto con_func_x_x3_num_x1 = (con_func_x_num_x3_x1 * con_func_x_den
			+ 2 * phi_s * torch::outer(con_func_x_num_x3.squeeze(), phi_s_x1.squeeze())
			- 2 * phi_s * torch::outer(con_func_x_num_x1.squeeze(), phi_s_x3.squeeze())
			- 2 * con_func_x_num * torch::outer(phi_s_x3.squeeze(), phi_s_x1.squeeze())
			- 2 * con_func_x_num * phi_s * phi_s_x3_x1);
		auto con_func_x_x3_den_x1 = 2 * con_func_x_den * con_func_x_den_x1;
		auto con_func_x_x3_x1 = (con_func_x_x3_num_x1 * con_func_x_x3_den
			- torch::outer(con_func_x_x3_num.squeeze(), con_func_x_x3_den_x1.squeeze()))
			/ torch::pow(con_func_x_x3_den, 2);

		auto con_func_x1_x1 = -2 * con_func * (1 - torch::pow(con_func, 2))
			* torch::outer(con_func_x_x1.squeeze(), con_func_x_x1.squeeze())
			+ (1 - torch::pow(con_func, 2) * con_func_x_x1_x1);

		auto con_func_x3_x3 = -2 * con_func * (1 - torch::pow(con_func, 2))
			* torch::outer(con_func_x_x3.squeeze(), con_func_x_x3.squeeze())
			+ (1 - torch::pow(con_func, 2) * con_func_x_x3_x3);

		auto con_func_x1_x3 = -2 * con_func * (1 - torch::pow(con_func, 2))
			* torch::outer(con_func_x_x3.squeeze(), con_func_x_x1.squeeze())
			+ (1 - torch::pow(con_func, 2)) * con_func_x_x1_x3;

		auto con_func_x3_x1 = -2 * con_func * (1 - torch::pow(con_func, 2))
			* torch::outer(con_func_x_x1.squeeze(), con_func_x_x3.squeeze())
			+ (1 - torch::pow(con_func, 2)) * con_func_x_x3_x1;

		auto kx_x1_x1_term1 = c * (c - 1) * torch::pow(gamma, c - 2)
			* torch::outer(gamma_x1.squeeze(), gamma_x1.squeeze())
			+ c * torch::pow(gamma, c - 1) * gamma_x1_x1;
		auto kx_x1_x1_term2 = c * (c - 1) * torch::pow(gamma, c - 2)
			* torch::outer(gamma_x1.squeeze(), gamma_x1.squeeze()) * con_func
			+ c * torch::pow(gamma, c - 1) * gamma_x1_x1 * con_func
			+ c * torch::pow(gamma, c - 1) * torch::outer(gamma_x1.squeeze(), con_func_x1.squeeze());
		auto kx_x1_x1_term3 = -c * torch::pow(gamma, c - 1)
			* torch::outer(gamma_x1.squeeze(), con_func_x1.squeeze())
			+ (1 - torch::pow(gamma, c)) * con_func_x1_x1;

		auto kx_x3_x3_term1 = c * (c - 1) * torch::pow(gamma, c - 2)
			* torch::outer(gamma_x3.squeeze(), gamma_x3.squeeze())
			+ c * torch::pow(gamma, c - 1) * gamma_x3_x3;
		auto kx_x3_x3_term2 = c * (c - 1) * torch::pow(gamma, c - 2)
			* torch::outer(gamma_x3.squeeze(), gamma_x3.squeeze()) * con_func
			+ c * torch::pow(gamma, c - 1) * gamma_x3_x3 * con_func
			+ c * torch::pow(gamma, c - 1)
			* torch::outer(gamma_x3.squeeze(), con_func_x3.squeeze());
		auto kx_x3_x3_term3 = -c * torch::pow(gamma, c - 1)
			* torch::outer(gamma_x3.squeeze(), con_func_x3.squeeze())
			+ (1 - torch::pow(gamma, c)) * con_func_x3_x3;

		auto kx_x1_x3_term1 = c * (c - 1) * torch::pow(gamma, c - 2)
			* torch::outer(gamma_x1.squeeze(), gamma_x3.squeeze())
			+ c * torch::pow(gamma, c - 1) * gamma_x1_x3;
		auto kx_x1_x3_term2 = c * (c - 1) * torch::pow(gamma, c - 2)
			* torch::outer(gamma_x1.squeeze(), gamma_x3.squeeze()) * con_func
			+ c * torch::pow(gamma, c - 1) * gamma_x1_x3 * con_func
			+ c * torch::pow(gamma, c - 1)
			* torch::outer(gamma_x1.squeeze(), con_func_x3.squeeze());
		auto kx_x1_x3_term3 = -c * torch::pow(gamma, c - 1)
			* torch::outer(con_func_x1.squeeze(), gamma_x3.squeeze())
			+ (1 - torch::pow(gamma, c)) * con_func_x1_x3;

		auto kx_x3_x1_term1 = c * (c - 1) * torch::pow(gamma, c - 2)
			* torch::outer(gamma_x3.squeeze(), gamma_x1.squeeze())
			+ c * torch::pow(gamma, c - 1) * gamma_x3_x1;
		auto kx_x3_x1_term2 = c * (c - 1) * torch::pow(gamma, c - 2)
			* torch::outer(gamma_x3.squeeze(), gamma_x1.squeeze()) * con_func
			+ c * torch::pow(gamma, c - 1) * gamma_x3_x1 * con_func
			+ c * torch::pow(gamma, c - 1) * torch::outer(gamma_x3.squeeze(), con_func_x1.squeeze());
		auto kx_x3_x1_term3 = -c * torch::pow(gamma, c - 1)
			* torch::outer(con_func_x3.squeeze(), gamma_x1.squeeze())
			+ (1 - torch::pow(gamma, c)) * con_func_x3_x1;

		kx = 0.5 * (comp_shear + 1.0) * env->cloth->S * PI * pow(env->cloth->R, 2)
			* ((1 + torch::pow(gamma, c)) + (1 - torch::pow(gamma, c)) * con_func);

		auto kx_x1 = 0.5 * (comp_shear + 1.0) * env->cloth->S * PI * pow(env->cloth->R, 2) *
			(c * torch::pow(gamma, c - 1) * gamma_x1 - c * torch::pow(gamma, c - 1) * gamma_x1 * con_func
				+ (1 - torch::pow(gamma, c)) * (1 - torch::pow(con_func, 2) * con_func_x_x1));

		auto kx_x3 = 0.5 * (comp_shear + 1.0) * env->cloth->S * PI * pow(env->cloth->R, 2) *
			(c * torch::pow(gamma, c - 1) * gamma_x3 - c * torch::pow(gamma, c - 1) * gamma_x3 * con_func
				+ (1 - torch::pow(gamma, c)) * (1 - torch::pow(con_func, 2) * con_func_x_x3));

		auto kx_x1_x1 = 0.5 * (comp_shear + 1.0) * env->cloth->S * pow(env->cloth->R, 2)
			* (kx_x1_x1_term1 - kx_x1_x1_term2 + kx_x1_x1_term3);

		auto kx_x3_x3 = 0.5 * (comp_shear + 1.0) * env->cloth->S * pow(env->cloth->R, 2)
			* (kx_x3_x3_term1 - kx_x3_x3_term2 + kx_x3_x3_term3);

		auto kx_x1_x3 = 0.5 * (comp_shear + 1.0) * env->cloth->S * pow(env->cloth->R, 2)
			* (kx_x1_x3_term1 - kx_x1_x3_term2 + kx_x1_x3_term3);

		auto kx_x3_x1 = 0.5 * (comp_shear + 1.0) * env->cloth->S * pow(env->cloth->R, 2)
			* (kx_x3_x1_term1 - kx_x3_x1_term2 + kx_x3_x1_term3);

		auto F_x1_x1_ori = ((kx * env->cloth->L) / (torch::pow(s.l1, 2) * torch::sin(s.phi)))
			* ((s.phi - phi_bar) * (-torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d1.squeeze())
				+ (torch::cos(s.phi) / torch::pow(torch::sin(s.phi), 2))
				* torch::mm(torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d3.squeeze()), s.P1)
				- torch::cos(s.phi) * s.P1 - torch::mm(torch::outer(s.d1.squeeze(), s.d3.squeeze()), s.P1))
				- (1.0 / torch::sin(s.phi)) * torch::mm(torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d3.squeeze()), s.P1));

		auto F_x3_x3_ori = ((kx * env->cloth->L) / (torch::pow(s.l3, 2) * torch::sin(s.phi)))
			* ((s.phi - phi_bar) * (-torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d3.squeeze())
				+ (torch::cos(s.phi) / torch::pow(torch::sin(s.phi), 2))
				* torch::mm(torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d1.squeeze()), s.P3)
				- torch::cos(s.phi) * s.P3 - torch::mm(torch::outer(s.d3.squeeze(), s.d1.squeeze()), s.P3))
				- (1.0 / torch::sin(s.phi)) * torch::mm(torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d1.squeeze()), s.P3));

		auto F_x1_x3_ori = ((kx * env->cloth->L) / (s.l1 * s.l3 * torch::sin(s.phi)))
			* ((s.phi - phi_bar) * ((torch::cos(s.phi) / torch::pow(torch::sin(s.phi), 2))
				* torch::mm(torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d1.squeeze()), s.P3) + torch::mm(s.P1, s.P3))
				- (1.0 / torch::sin(s.phi)) * torch::mm(torch::outer(torch::mm(s.P1, s.d3).squeeze(), s.d1.squeeze()), s.P3));

		auto F_x3_x1_ori = ((kx * env->cloth->L) / (s.l3 * s.l1 * torch::sin(s.phi)))
			* ((s.phi - phi_bar) * ((torch::cos(s.phi) / torch::pow(torch::sin(s.phi), 2))
				* torch::mm(torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d3.squeeze()), s.P1) + torch::mm(s.P3, s.P1))
				- (1.0 / torch::sin(s.phi)) * torch::mm(torch::outer(torch::mm(s.P3, s.d1).squeeze(), s.d3.squeeze()), s.P1));

		auto F_x1_x1 = -0.5 * kx_x1_x1 * env->cloth->L * torch::pow(s.phi - phi_bar, 2)
			- 2 * env->cloth->L * (s.phi - phi_bar) * torch::outer(kx_x1.squeeze(), phi_x1.squeeze()) + F_x1_x1_ori;

		auto F_x3_x3 = -0.5 * kx_x3_x3 * env->cloth->L * torch::pow(s.phi - phi_bar, 2)
			- 2 * env->cloth->L * (s.phi - phi_bar) * torch::outer(kx_x3.squeeze(), phi_x3.squeeze()) + F_x3_x3_ori;

		auto F_x1_x3 = (-0.5 * kx_x1_x3 * env->cloth->L * torch::pow(s.phi - phi_bar, 2)
			- env->cloth->L * (s.phi - phi_bar) * torch::outer(kx_x1.squeeze(), phi_x3.squeeze())
			- env->cloth->L * (s.phi - phi_bar) * torch::outer(phi_x1.squeeze(), kx_x3.squeeze())
			+ F_x1_x3_ori);

		auto F_x3_x1 = (-0.5 * kx_x3_x1 * env->cloth->L * torch::pow(s.phi - phi_bar, 2)
			- env->cloth->L * (s.phi - phi_bar) * torch::outer(kx_x3.squeeze(), phi_x1.squeeze())
			- env->cloth->L * (s.phi - phi_bar) * torch::outer(phi_x3.squeeze(), kx_x1.squeeze())
			+ F_x3_x1_ori);

		auto F_x1_x0 = -(F_x1_x1 + F_x1_x3);
		auto F_x3_x0 = -(F_x3_x1 + F_x3_x3);
		auto F_x0_x1 = -(F_x1_x1 + F_x3_x1);
		auto F_x0_x3 = -(F_x1_x3 + F_x3_x3);
		auto F_x0_x0 = -(F_x1_x0 + F_x3_x0);

		F_LL_Pos[L0L0_idx] = F_LL_Pos[L0L0_idx] + F_x0_x0;
		F_LL_Pos[L0L1_idx] = F_LL_Pos[L0L1_idx] + F_x0_x1;
		F_LL_Pos[L0L3_idx] = F_LL_Pos[L0L3_idx] + F_x0_x3;
		F_LL_Pos[L1L0_idx] = F_LL_Pos[L1L0_idx] + F_x1_x0;
		F_LL_Pos[L1L1_idx] = F_LL_Pos[L1L1_idx] + F_x1_x1;
		F_LL_Pos[L1L3_idx] = F_LL_Pos[L1L3_idx] + F_x1_x3;
		F_LL_Pos[L3L0_idx] = F_LL_Pos[L3L0_idx] + F_x3_x0;
		F_LL_Pos[L3L1_idx] = F_LL_Pos[L3L1_idx] + F_x3_x1;
		F_LL_Pos[L3L3_idx] = F_LL_Pos[L3L3_idx] + F_x3_x3;

		return;
	}

	Tensor L1L1_Shear = ((kx * env->cloth->L) / (pow(s.l1, 2) * torch::sin(s.phi))) * ((s.phi - PI / 2) *
		(-1 * torch::mm(torch::mm(s.P1, s.d3), s.d1.transpose(1, 0)) + (torch::cos(s.phi) / pow(torch::sin(s.phi), 2)) *
			torch::mm(torch::mm(torch::mm(s.P1, s.d3), s.d3.transpose(1, 0)), s.P1) - torch::cos(s.phi) * s.P1 -
			torch::mm(torch::mm(s.d1, s.d3.transpose(1, 0)), s.P1)) -
		(1 / torch::sin(s.phi)) * torch::mm(torch::mm(torch::mm(s.P1, s.d3), s.d3.transpose(1, 0)), s.P1));

	Tensor L1L3_Shear = ((kx * env->cloth->L) / (s.l3 * s.l1 * torch::sin(s.phi))) * torch::mm(((s.phi - PI / 2) * ((torch::cos(s.phi) / pow(torch::sin(s.phi), 2)) * torch::mm(torch::mm(s.P1, s.d3), s.d1.transpose(1, 0))
		- (1 / torch::sin(s.phi)) * torch::mm(torch::mm(s.P1, s.d3), s.d1.transpose(1, 0)))), s.P3);

	Tensor L3L1_Shear = ((kx * env->cloth->L) / (s.l1 * s.l3 * torch::sin(s.phi))) * torch::mm(((s.phi - PI / 2) * ((torch::cos(s.phi) / pow(torch::sin(s.phi), 2)) * torch::mm(torch::mm(s.P3, s.d1), s.d3.transpose(1, 0))
		- (1 / torch::sin(s.phi)) * torch::mm(torch::mm(s.P3, s.d1), s.d3.transpose(1, 0)))), s.P1);

	Tensor L3L3_Shear = ((kx * env->cloth->L) / (pow(s.l3, 2) * torch::sin(s.phi))) * ((s.phi - PI / 2) *
		(-1 * torch::mm(torch::mm(s.P3, s.d1), s.d3.transpose(1, 0)) + (torch::cos(s.phi) / pow(torch::sin(s.phi), 2)) *
			torch::mm(torch::mm(torch::mm(s.P3, s.d1), s.d1.transpose(1, 0)), s.P3) - torch::cos(s.phi) * s.P3 -
			torch::mm(torch::mm(s.d3, s.d1.transpose(1, 0)), s.P3)) -
		(1 / torch::sin(s.phi)) * torch::mm(torch::mm(torch::mm(s.P3, s.d1), s.d1.transpose(1, 0)), s.P3));

	Tensor L1L0_Shear = -(L1L1_Shear + L1L3_Shear);
	Tensor L3L0_Shear = -(L3L1_Shear + L3L3_Shear);
	Tensor L0L1_Shear = -(L1L1_Shear + L3L1_Shear);
	Tensor L0L3_Shear = -(L1L3_Shear + L3L3_Shear);
	Tensor L0L0_Shear = -(L1L0_Shear + L3L0_Shear);

	F_LL_Pos[L0L0_idx] = F_LL_Pos[L0L0_idx] + L0L0_Shear;
	F_LL_Pos[L0L1_idx] = F_LL_Pos[L0L1_idx] + L0L1_Shear;
	F_LL_Pos[L0L3_idx] = F_LL_Pos[L0L3_idx] + L0L3_Shear;
	F_LL_Pos[L1L0_idx] = F_LL_Pos[L1L0_idx] + L1L0_Shear;
	F_LL_Pos[L1L1_idx] = F_LL_Pos[L1L1_idx] + L1L1_Shear;
	F_LL_Pos[L1L3_idx] = F_LL_Pos[L1L3_idx] + L1L3_Shear;
	F_LL_Pos[L3L0_idx] = F_LL_Pos[L3L0_idx] + L3L0_Shear;
	F_LL_Pos[L3L1_idx] = F_LL_Pos[L3L1_idx] + L3L1_Shear;
	F_LL_Pos[L3L3_idx] = F_LL_Pos[L3L3_idx] + L3L3_Shear;
}

void PhysicsCloth::Wind_F(const Face& f, Tensor& F_L)
{
	int F1_idx{ f.n1->idx_u + f.n1->idx_v * env->cloth->width };
	int F2_idx{ f.n2->idx_u + f.n2->idx_v * env->cloth->width };
	int F3_idx{ f.n3->idx_u + f.n3->idx_v * env->cloth->width };

	Tensor face_vel{ (f.n1->LVel + f.n2->LVel + f.n3->LVel) / 3.0 };
	Tensor rela_vel{ env->wind.WindVelocity - face_vel };
	Tensor norm_vel{ torch::mm(f.n.transpose(0,1), rela_vel) };
	Tensor tang_vel{ rela_vel - norm_vel * f.n };
	Tensor wind_force{ env->wind.WindDensity * f.area * norm_vel.abs() * norm_vel * f.n + env->wind.WindDrag * f.area * tang_vel };

	F_L[F1_idx] = F_L[F1_idx] + wind_force / 3.0;
	F_L[F2_idx] = F_L[F2_idx] + wind_force / 3.0;
	F_L[F3_idx] = F_L[F3_idx] + wind_force / 3.0;

}