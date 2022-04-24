#pragma once
#include "TrainHelper.h"

struct YarnSimNetHomo : torch::nn::Module {
	YarnSimNetHomo(H5ClothSize size, H5Physical phy,
		std::vector<std::pair<int, int>> _handles, Tensor _h,
		std::pair<double, double> _rho_range, std::pair<double, double> _Y_range,
		std::pair<double, double> _B_range, std::pair<double, double> _S_range) :
		width(size.width), length(size.length), height(0.0), R(phy.R), L(phy.L) {

		G = torch::tensor({ 0.0, 0.0, phy.gravity }, opts).view({ 3,1 });

		rho = register_parameter("rho", torch::tensor({ phy.rho }, opts));
		Y = register_parameter("Y", torch::tensor({ phy.Y }, opts));
		B = register_parameter("B", torch::tensor({ phy.B }, opts));
		S = register_parameter("S", torch::tensor({ phy.S }, opts));

		kf = torch::tensor({ phy.kf }, opts);
		df = torch::tensor({ phy.df }, opts);
		mu = torch::tensor({ phy.mu }, opts);
		kc = torch::tensor({ phy.kc }, opts);

		handle_stiff = torch::tensor({ phy.handle_stiffness }, opts);

		Tensor wind_vel = torch::tensor({ phy.wind_velocity_x, phy.wind_velocity_y ,phy.wind_velocity_z }, opts).view({ 3,1 });
		Tensor wind_density = torch::tensor({ phy.wind_density }, opts);
		Tensor wind_drag = torch::tensor({ phy.wind_drag }, opts);

		wind = Wind(wind_vel, wind_density, wind_drag);
		
		h = _h;

		handles = _handles;

		woven_pattern = WovenPattern(phy.woven_pattern);

		rho_range = _rho_range;
		Y_range = _Y_range;
		B_range = _B_range;
		S_range = _S_range;

		Tensor rho_constraint = constraint_para(rho, rho_range.first, rho_range.second);
		Tensor Y_constraint = constraint_para(Y, Y_range.first, Y_range.second);
		Tensor B_constraint = constraint_para(B, B_range.first, B_range.second);
		Tensor S_constraint = constraint_para(S, S_range.first, S_range.second);

		Yarn yarn{ R, rho_constraint, Y_constraint, B_constraint };

		yarns = std::vector{ yarn };
	}

	void reset_cloth() {
		Tensor rho_constraint = constraint_para(rho, rho_range.first, rho_range.second);
		Tensor Y_constraint = constraint_para(Y, Y_range.first, Y_range.second);
		Tensor B_constraint = constraint_para(B, B_range.first, B_range.second);
		Tensor S_constraint = constraint_para(S, S_range.first, S_range.second);

		yarns[0] = Yarn(R, rho_constraint, Y_constraint, B_constraint);

		cloth = Cloth(width, length, height, L, R, yarns, kc, kf, df, mu, S, handle_stiff, woven_pattern, InitPose::Upright);

		cloth.set_handles(handles);

		env = Environment(&cloth, G, wind);

		phy_sim = PhysicsCloth(&env);
		phy_sim.InitMV();
	}

	ClothState forward(ClothState cloth_state, bool& is_success) {
		cloth.set_ClothState(cloth_state);
		is_success = SimStep(cloth, phy_sim, h);
		return cloth.get_ClothState();
	}

	ClothState forward(bool& is_success) {
		is_success = SimStep(cloth, phy_sim, h);
		return cloth.get_ClothState();
	}

	ClothState forward(ClothState cloth_state, bool& is_success, std::string& render_path) {
		cloth.set_ClothState(cloth_state);
		is_success = SimStep(cloth, phy_sim, h);
		if (is_success)
			RenderCloth(cloth, render_path, RenderType::MeshTriangle);
		return cloth.get_ClothState();
	}

	ClothState forward(bool& is_success, std::string& render_path) {
		is_success = SimStep(cloth, phy_sim, h);
		if (is_success)
			RenderCloth(cloth, render_path, RenderType::MeshTriangle);
		return cloth.get_ClothState();
	}

	int width, length;
	double height, R, L;
	Tensor rho, G, Y, B, S, kf, df, mu, kc;
	Tensor handle_stiff;
	Wind wind;
	WovenPattern woven_pattern;
	Tensor h;
	std::pair<double, double> rho_range;
	std::pair<double, double> Y_range;
	std::pair<double, double> B_range;
	std::pair<double, double> S_range;
	std::vector<std::pair<int, int>> handles;
	std::vector<Yarn> yarns;
	Cloth cloth;
	Environment env;
	PhysicsCloth phy_sim;
};