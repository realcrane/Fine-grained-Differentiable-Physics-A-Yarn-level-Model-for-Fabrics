#pragma once

#include "MatrixMp.h"
#include "Environment.h"

struct PhysicsCloth
{
	Environment* env;

	std::vector<Tensor> Pos_L, Pos_E;
	std::vector<Tensor> Vel_L, Vel_E;

	Tensor M_LLs[NThrd], M_ELs[NThrd], M_LEs[NThrd], M_EEs[NThrd];
	Tensor F_Ls[NThrd], F_Es[NThrd];
	Tensor F_LL_Poss[NThrd], F_EL_Poss[NThrd], F_LE_Poss[NThrd], F_EE_Poss[NThrd];
	Tensor F_LL_Vels[NThrd], F_EL_Vels[NThrd], F_LE_Vels[NThrd], F_EE_Vels[NThrd];
	Tensor StrhBend_NC_Ls[NThrd], StrhBend_C_Ls[NThrd];
	Tensor StrhBend_NC_LEs[NThrd], StrhBend_C_LEs[NThrd];
	Tensor F_E_is_Slides[NThrd];
	Tensor F_E_No_Slide, F_EE_Pos_No_Slide, F_EE_Vel_No_Slide;

	PhysicsCloth() :env{ nullptr } {}	// Default Constructor
	PhysicsCloth(Environment* _env) : env(_env) {}

	int find_index(int F_u, int F_v, int D_u, int D_v, std::string D_type, std::string uv);

	void InitMV();	// Initialize Matrix and Vector
	void FillMV();	// Fill Matrix and Vector
	void ResetMV();	// Reset Matrix and Vector

	void VecPosVel(int u, int v);

	void Mass(const Edge& e, Tensor& M_LL, Tensor& M_EL, Tensor& M_LE, Tensor& M_EE);

	void Gravity_F(const Edge& e, Tensor& F_L, Tensor& F_E);
	void Gravity_D(const Edge& e, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos);

	void Stretch_F(const Edge& e, Tensor& F_L, Tensor& F_E, Tensor& StrhBend_NC_L, Tensor& StrhBend_C_L);
	void Stretch_D(const Edge& e, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos, Tensor& StrhBend_NC_LE, Tensor& StrhBend_C_LE);

	void Bending_F(const Bend_Seg& b, Tensor& F_L, Tensor& F_E, Tensor& StrhBend_NC_L);
	void Bending_D(const Bend_Seg& b, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos, Tensor& StrhBend_NC_LE);

	void Crimp_Bending_F(const Bend_Seg& b, Tensor& StrhBend_C_L);
	void Crimp_Bending_D(const Bend_Seg& b, Tensor& StrhBend_C_LE);

	void ParaColli_F(const Edge& e, Tensor& F_E);
	void ParaColli_D(const Edge& e, Tensor& F_EE_Pos);

	void Inertia_F(const Edge& e, Tensor& F_L, Tensor& F_E);
	void Inertia_D_Pos(const Edge& e, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos);
	void Inertia_D_Vel(const Edge& e, Tensor& F_LL_Vel, Tensor& F_EL_Vel, Tensor& F_LE_Vel, Tensor& F_EE_Vel);

	void MDotVDot_F(const Edge& e, Tensor& F_L, Tensor& F_E);
	void MDotVDot_D_Pos(const Edge& e, Tensor& F_LL_Pos, Tensor& F_EL_Pos, Tensor& F_LE_Pos, Tensor& F_EE_Pos);
	void MDotVDot_D_Vel(const Edge& e, Tensor& F_LL_Vel, Tensor& F_EL_Vel, Tensor& F_LE_Vel, Tensor& F_EE_Vel);

	void Constraint_F(int u, int v, Tensor& F_L);
	void Constraint_D(int u, int v, Tensor& F_LL_Pos);

	void ConSlideFriction(Node& n, Tensor& F_E_is_Slide,
		Tensor& F_E_No_Slide, Tensor& F_EE_Pos_No_Slide, Tensor& F_EE_Vel_No_Slide,
		Tensor& F_E, Tensor& F_EE_Pos, Tensor& F_EE_Vel,
		Tensor& StrhBend_C_L, Tensor& StrhBend_NC_L,
		Tensor& StrhBend_C_LE, Tensor& StrhBend_NC_LE);

	void Shear_F(const Shear_Seg& s, Tensor& F_L);
	void Shear_D(const Shear_Seg& s, Tensor& F_LL_Pos);

	void Wind_F(const Face& f, Tensor& F_L);

	Tensor get_SpaM() const { return To_Sparse_Mat(env->cloth->width, env->cloth->length, M_LLs[0], M_LEs[0], M_ELs[0], M_EEs[0]); }
	Tensor get_SpaF() const { return To_Sparse_Vec(F_Ls[0], F_Es[0]); }
	Tensor get_SpaFPos() const { return To_Sparse_Mat(env->cloth->width, env->cloth->length, F_LL_Poss[0], F_LE_Poss[0], F_EL_Poss[0], F_EE_Poss[0]); }
	Tensor get_SpaFVel() const { return To_Sparse_Mat(env->cloth->width, env->cloth->length, F_LL_Vels[0], F_LE_Vels[0], F_EL_Vels[0], F_EE_Vels[0]); }
	Tensor get_SpaPos() const { return To_Sparse_Vec(Pos_L, Pos_E); }
	Tensor get_SpaVel() const { return To_Sparse_Vec(Vel_L, Vel_E); }
};