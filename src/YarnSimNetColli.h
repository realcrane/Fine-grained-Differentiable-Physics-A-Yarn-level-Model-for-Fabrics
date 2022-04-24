#pragma once
#include "TrainHelper.h"

struct YarnSimNetColli : torch::nn::Module {
	YarnSimNetColli(H5ClothSize size, double _height, H5Physical phy,
		std::vector<std::pair<int, int>> _handles, Tensor _h, bool _is_para,
		std::pair<double, double> _rho_range, std::pair<double, double> _Y_range,
		std::pair<double, double> _B_range, std::string ground_path, std::string table_path) :
		width(size.width), length(size.length), height(_height), R(phy.R), L(phy.L) {
		
		G = torch::tensor({ 0.0, 0.0, phy.gravity }, opts).view({ 3,1 });
		
		rho = register_parameter("rho", torch::tensor({ phy.rho }, opts));
		Y = register_parameter("Y", torch::tensor({ phy.Y }, opts));
		B = register_parameter("B", torch::tensor({ phy.B }, opts));
		
		S = torch::tensor({ phy.S }, opts);
		kf = torch::tensor({ phy.kf }, opts);
		df = torch::tensor({ phy.df }, opts);
		mu = torch::tensor({ phy.mu }, opts);
		kc = torch::tensor({ phy.kc }, opts);

		handle_stiff = torch::tensor({ phy.handle_stiffness }, opts);

		Tensor wind_vel = torch::tensor({ phy.wind_velocity_x, phy.wind_velocity_y ,phy.wind_velocity_z }, opts).view({ 3,1 });
		Tensor wind_density = torch::tensor({ phy.wind_density }, opts);
		Tensor wind_drag = torch::tensor({ phy.wind_drag }, opts);

		wind = Wind(wind_vel, wind_density, wind_drag);

		handles = _handles;
		thickness = torch::tensor({ 0.0005 }, opts);
		h = _h;
		is_para = _is_para;

		woven_pattern = WovenPattern(phy.woven_pattern);

		rho_range = _rho_range;
		Y_range = _Y_range;
		B_range = _B_range;

		table.obsMesh.isCloth = false;
		ground.obsMesh.isCloth = false;

		table.density = torch::tensor({ 1e3 }, opts);
		ground.density = torch::tensor({ 1e3 }, opts);

		load_obj(table_path, table.obsMesh);
		load_obj(ground_path, ground.obsMesh);

		table.obsMesh.add_adjecent();
		table.compute_mesh_mass();

		ground.obsMesh.add_adjecent();
		ground.compute_mesh_mass();

		Tensor rho_constraint = constraint_para(rho, rho_range.first, rho_range.second);
		Tensor Y_constraint = constraint_para(Y, Y_range.first, Y_range.second);
		Tensor B_constraint = constraint_para(B, B_range.first, B_range.second);

		Yarn yarn{R, rho_constraint, Y_constraint, B_constraint};

		yarns = std::vector{yarn};
	}

	void reset_cloth() {

		Tensor rho_constraint = constraint_para(rho, rho_range.first, rho_range.second);
		Tensor Y_constraint = constraint_para(Y, Y_range.first, Y_range.second);
		Tensor B_constraint = constraint_para(B, B_range.first, B_range.second);

		yarns[0] = Yarn(R, rho_constraint, Y_constraint, B_constraint);

		cloth = Cloth(width, length, height, L, R, yarns, kc, kf, df, mu, S, handle_stiff, WovenPattern::Plain, InitPose::Upright);

		cloth.set_handles(handles);

		env = Environment(&cloth, G, wind);

		phy_sim = PhysicsCloth(&env);

		phy_sim.InitMV();

		collision = Collision(&cloth.clothMesh, &table.obsMesh, &ground.obsMesh, thickness, h, true);
	}

	ClothState forward(ClothState cloth_state, bool& is_success) {
		cloth.set_ClothStateMesh(cloth_state);
		is_success = SimStep(cloth, phy_sim, collision, h, is_para);
		return cloth.get_ClothState();
	}

	ClothState forward(bool& is_success) {
		is_success = SimStep(cloth, phy_sim, collision, h, is_para);
		return cloth.get_ClothState();
	}

	int width, length;
	double height, R, L;
	Tensor rho, G, Y, B, S, kf, df, mu, kc;
	Tensor handle_stiff;
	Wind wind;
	WovenPattern woven_pattern;
	Tensor h;
	Tensor thickness;
	bool is_para;
	std::pair<double, double> rho_range;
	std::pair<double, double> Y_range;
	std::pair<double, double> B_range;
	std::vector<std::pair<int, int>> handles;
	Obstacle table, ground;
	std::vector<Yarn> yarns;
	Cloth cloth;
	Collision collision;
	Environment env;
	PhysicsCloth phy_sim;
};