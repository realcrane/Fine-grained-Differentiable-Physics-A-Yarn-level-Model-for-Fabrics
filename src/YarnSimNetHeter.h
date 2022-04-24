#pragma once
#include "TrainHelper.h"

struct YarnSimNetHeter : torch::nn::Module {
	YarnSimNetHeter(H5ClothSize size, H5PhysicalHeterYarn phy,
		std::vector<std::pair<int, int>> _handles, Tensor _h,
		std::pair<double, double> _rho1_range, std::pair<double, double> _rho2_range,
		std::pair<double, double> _Y1_range, std::pair<double, double> _Y2_range,
		std::pair<double, double> _B1_range, std::pair<double, double> _B2_range,
		std::pair<double, double> _S_range, std::pair<double, double> _mu_range) :
		width(size.width), length(size.length), hight(5.0), R(phy.R), L(phy.L) {

		G = torch::tensor({ 0.0, 0.0, phy.gravity }, opts).view({ 3,1 });

		rho1 = register_parameter("rho_1", torch::tensor({ phy.rho1 }, opts));
		rho2 = register_parameter("rho_2", torch::tensor({ phy.rho2 }, opts));

		Y1 = register_parameter("Y_1", torch::tensor({ phy.Y1 }, opts));
		Y2 = register_parameter("Y_2", torch::tensor({ phy.Y2 }, opts));

		B1 = register_parameter("B_1", torch::tensor({ phy.B1 }, opts));
		B2 = register_parameter("B_2", torch::tensor({ phy.B2 }, opts));

		S = register_parameter("S", torch::tensor({ phy.S }, opts));
		mu = register_parameter("mu", torch::tensor({ phy.mu }, opts));
		
		kf = torch::tensor({ phy.kf }, opts);
		df = torch::tensor({ phy.df }, opts);
		kc = torch::tensor({ phy.kc }, opts);
		
		handle_stiff = torch::tensor({ phy.handle_stiffness }, opts);

		Tensor wind_vel = torch::tensor({ phy.wind_velocity_x, phy.wind_velocity_y ,phy.wind_velocity_z }, opts).view({ 3,1 });
		Tensor wind_density = torch::tensor({ phy.wind_density }, opts);
		Tensor wind_drag = torch::tensor({ phy.wind_drag }, opts);

		wind = Wind(wind_vel, wind_density, wind_drag);

		handles = _handles;
		h = _h;
		woven_pattern = WovenPattern(phy.woven_pattern);

		rho1_range = _rho1_range;
		rho2_range = _rho2_range;

		Y1_range = _Y1_range;
		Y2_range = _Y2_range;

		B1_range = _B1_range;
		B2_range = _B2_range;

		S_range = _S_range;

		mu_range = _mu_range;

		Tensor rho1_constraint = constraint_para(rho1, rho1_range.first, rho1_range.second);
		Tensor rho2_constraint = constraint_para(rho2, rho2_range.first, rho2_range.second);

		Tensor Y1_constraint = constraint_para(Y1, Y1_range.first, Y1_range.second);
		Tensor Y2_constraint = constraint_para(Y2, Y2_range.first, Y2_range.second);

		Tensor B1_constraint = constraint_para(B1, B1_range.first, B1_range.second);
		Tensor B2_constraint = constraint_para(B2, B2_range.first, B2_range.second);

		Tensor S_constraint = constraint_para(S, S_range.first, S_range.second);

		Tensor mu_constraint = constraint_para(mu, mu_range.first, mu_range.second);

		yarn1 = Yarn{ R, rho1_constraint, Y1_constraint, B1_constraint };
		yarn2 = Yarn{ R, rho2_constraint, Y2_constraint, B2_constraint };

		yarns = std::vector{ yarn1, yarn2 };
	}

	void reset_cloth() {
		Tensor rho1_constraint = constraint_para(rho1, rho1_range.first, rho1_range.second);
		Tensor rho2_constraint = constraint_para(rho2, rho2_range.first, rho2_range.second);

		Tensor Y1_constraint = constraint_para(Y1, Y1_range.first, Y1_range.second);
		Tensor Y2_constraint = constraint_para(Y2, Y2_range.first, Y2_range.second);

		Tensor B1_constraint = constraint_para(B1, B1_range.first, B1_range.second);
		Tensor B2_constraint = constraint_para(B2, B2_range.first, B2_range.second);

		Tensor S_constraint = constraint_para(S, S_range.first, S_range.second);

		Tensor mu_constraint = constraint_para(mu, mu_range.first, mu_range.second);

		yarns[0] = Yarn{ R, rho1_constraint, Y1_constraint, B1_constraint };
		yarns[1] = Yarn{ R, rho2_constraint, Y2_constraint, B2_constraint };

		cloth = Cloth(width, length, hight, L, R, yarns, kc, kf, df, mu_constraint, S_constraint, handle_stiff, woven_pattern, InitPose::Upright);

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

	int width, length;
	double hight, R, L;
	Tensor rho1, rho2;
	Tensor Y1, Y2;
	Tensor B1, B2;
	Tensor G, S, kf, df, mu, kc;
	Tensor handle_stiff;
	Wind wind;
	Yarn yarn1, yarn2;
	std::vector<Yarn> yarns;
	WovenPattern woven_pattern;
	Tensor h;
	std::pair<double, double> rho1_range, rho2_range;
	std::pair<double, double> Y1_range, Y2_range;
	std::pair<double, double> B1_range, B2_range;
	std::pair<double, double> S_range;
	std::pair<double, double> mu_range;
	std::vector<std::pair<int, int>> handles;
	Cloth cloth;
	Environment env;
	PhysicsCloth phy_sim;
};

struct YarnSimNetHeterMid : torch::nn::Module {
	YarnSimNetHeterMid(H5ClothSize size, H5PhysicalHeterYarn phy,
		std::vector<std::pair<int, int>> _handles, Tensor _h,
		std::pair<double, double> _rho1_range, std::pair<double, double> _rho2_range,
		std::pair<double, double> _Y1_range, std::pair<double, double> _Y2_range,
		std::pair<double, double> _B1_range, std::pair<double, double> _B2_range,
		std::pair<double, double> _S_range) :
		width(size.width), length(size.length), hight(5.0), R(phy.R), L(phy.L) {

		G = torch::tensor({ 0.0, 0.0, phy.gravity }, opts).view({ 3,1 });

		rho1 = register_parameter("rho_1", torch::tensor({ phy.rho1 }, opts));
		rho2 = register_parameter("rho_2", torch::tensor({ phy.rho2 }, opts));

		Y1 = register_parameter("Y_1", torch::tensor({ phy.Y1 }, opts));
		Y2 = register_parameter("Y_2", torch::tensor({ phy.Y2 }, opts));

		B1 = register_parameter("B_1", torch::tensor({ phy.B1 }, opts));
		B2 = register_parameter("B_2", torch::tensor({ phy.B2 }, opts));

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

		handles = _handles;
		h = _h;
		woven_pattern = WovenPattern(phy.woven_pattern);

		rho1_range = _rho1_range;
		rho2_range = _rho2_range;

		Y1_range = _Y1_range;
		Y2_range = _Y2_range;

		B1_range = _B1_range;
		B2_range = _B2_range;

		S_range = _S_range;

		Tensor rho1_constraint = constraint_para(rho1, rho1_range.first, rho1_range.second);
		Tensor rho2_constraint = constraint_para(rho2, rho2_range.first, rho2_range.second);

		Tensor Y1_constraint = constraint_para(Y1, Y1_range.first, Y1_range.second);
		Tensor Y2_constraint = constraint_para(Y2, Y2_range.first, Y2_range.second);

		Tensor B1_constraint = constraint_para(B1, B1_range.first, B1_range.second);
		Tensor B2_constraint = constraint_para(B2, B2_range.first, B2_range.second);

		Tensor S_constraint = constraint_para(S, S_range.first, S_range.second);

		yarn1 = Yarn{ R, rho1_constraint, Y1_constraint, B1_constraint };
		yarn2 = Yarn{ R, rho2_constraint, Y2_constraint, B2_constraint };

		yarns = std::vector{ yarn1, yarn2 };
	}

	void reset_cloth() {
		Tensor rho1_constraint = constraint_para(rho1, rho1_range.first, rho1_range.second);
		Tensor rho2_constraint = constraint_para(rho2, rho2_range.first, rho2_range.second);

		Tensor Y1_constraint = constraint_para(Y1, Y1_range.first, Y1_range.second);
		Tensor Y2_constraint = constraint_para(Y2, Y2_range.first, Y2_range.second);

		Tensor B1_constraint = constraint_para(B1, B1_range.first, B1_range.second);
		Tensor B2_constraint = constraint_para(B2, B2_range.first, B2_range.second);

		Tensor S_constraint = constraint_para(S, S_range.first, S_range.second);

		yarns[0] = Yarn{ R, rho1_constraint, Y1_constraint, B1_constraint };
		yarns[1] = Yarn{ R, rho2_constraint, Y2_constraint, B2_constraint };

		cloth = Cloth(width, length, hight, L, R, yarns, kc, kf, df, mu, S_constraint, handle_stiff, woven_pattern,  InitPose::Upright);

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

	int width, length;
	double hight, R, L;
	Tensor rho1, rho2;
	Tensor Y1, Y2;
	Tensor B1, B2;
	Tensor G, S, kf, df, mu, kc;
	Tensor handle_stiff;
	Wind wind;
	Yarn yarn1, yarn2;
	std::vector<Yarn> yarns;
	WovenPattern woven_pattern;
	Tensor h;
	std::pair<double, double> rho1_range, rho2_range;
	std::pair<double, double> Y1_range, Y2_range;
	std::pair<double, double> B1_range, B2_range;
	std::pair<double, double> S_range;;
	std::vector<std::pair<int, int>> handles;
	Cloth cloth;
	Environment env;
	PhysicsCloth phy_sim;
};

struct YarnSimNetHeterFew : torch::nn::Module {
	YarnSimNetHeterFew(H5ClothSize size, H5PhysicalHeterYarn phy,
		std::vector<std::pair<int, int>> _handles, Tensor _h,
		std::pair<double, double> _rho1_range, std::pair<double, double> _rho2_range,
		std::pair<double, double> _Y1_range, std::pair<double, double> _Y2_range,
		std::pair<double, double> _B1_range, std::pair<double, double> _B2_range) :
		width(size.width), length(size.length), hight(5.0), R(phy.R), L(phy.L) {

		G = torch::tensor({ 0.0, 0.0, phy.gravity }, opts).view({ 3,1 });

		rho1 = register_parameter("rho_1", torch::tensor({ phy.rho1 }, opts));
		rho2 = register_parameter("rho_2", torch::tensor({ phy.rho2 }, opts));

		Y1 = register_parameter("Y_1", torch::tensor({ phy.Y1 }, opts));
		Y2 = register_parameter("Y_2", torch::tensor({ phy.Y2 }, opts));

		B1 = register_parameter("B_1", torch::tensor({ phy.B1 }, opts));
		B2 = register_parameter("B_2", torch::tensor({ phy.B2 }, opts));

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
		h = _h;
		woven_pattern = WovenPattern(phy.woven_pattern);

		rho1_range = _rho1_range;
		rho2_range = _rho2_range;

		Y1_range = _Y1_range;
		Y2_range = _Y2_range;

		B1_range = _B1_range;
		B2_range = _B2_range;

		Tensor rho1_constraint = constraint_para(rho1, rho1_range.first, rho1_range.second);
		Tensor rho2_constraint = constraint_para(rho2, rho2_range.first, rho2_range.second);

		Tensor Y1_constraint = constraint_para(Y1, Y1_range.first, Y1_range.second);
		Tensor Y2_constraint = constraint_para(Y2, Y2_range.first, Y2_range.second);

		Tensor B1_constraint = constraint_para(B1, B1_range.first, B1_range.second);
		Tensor B2_constraint = constraint_para(B2, B2_range.first, B2_range.second);

		yarn1 = Yarn{ R, rho1_constraint, Y1_constraint, B1_constraint };
		yarn2 = Yarn{ R, rho2_constraint, Y2_constraint, B2_constraint };

		yarns = std::vector{ yarn1, yarn2 };
	}

	void reset_cloth() {
		Tensor rho1_constraint = constraint_para(rho1, rho1_range.first, rho1_range.second);
		Tensor rho2_constraint = constraint_para(rho2, rho2_range.first, rho2_range.second);

		Tensor Y1_constraint = constraint_para(Y1, Y1_range.first, Y1_range.second);
		Tensor Y2_constraint = constraint_para(Y2, Y2_range.first, Y2_range.second);

		Tensor B1_constraint = constraint_para(B1, B1_range.first, B1_range.second);
		Tensor B2_constraint = constraint_para(B2, B2_range.first, B2_range.second);

		yarns[0] = Yarn{ R, rho1_constraint, Y1_constraint, B1_constraint };
		yarns[1] = Yarn{ R, rho2_constraint, Y2_constraint, B2_constraint };

		cloth = Cloth(width, length, hight, L, R, yarns, kc, kf, df, mu, S, handle_stiff, woven_pattern, InitPose::Upright);

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

	int width, length;
	double hight, R, L;
	Tensor rho1, rho2;
	Tensor Y1, Y2;
	Tensor B1, B2;
	Tensor G, S, kf, df, mu, kc;
	Tensor handle_stiff;
	Wind wind;
	Yarn yarn1, yarn2;
	std::vector<Yarn> yarns;
	WovenPattern woven_pattern;
	Tensor h;
	std::pair<double, double> rho1_range, rho2_range;
	std::pair<double, double> Y1_range, Y2_range;
	std::pair<double, double> B1_range, B2_range;
	std::vector<std::pair<int, int>> handles;
	Cloth cloth;
	Environment env;
	PhysicsCloth phy_sim;
};

struct YarnSimNetHeterTest : torch::nn::Module {
	YarnSimNetHeterTest(H5ClothSize size, H5PhysicalHeterYarn phy,
		std::vector<std::pair<int, int>> _handles, Tensor _h, bool _is_para,
		std::pair<double, double> _rho1_range, std::pair<double, double> _rho2_range,
		std::pair<double, double> _Y1_range, std::pair<double, double> _Y2_range,
		std::pair<double, double> _B1_range, std::pair<double, double> _B2_range) :
		width(size.width), length(size.length), hight(5.0), R(phy.R), L(phy.L) {

		G = torch::tensor({ 0.0, 0.0, phy.gravity }, opts).view({ 3,1 });

		rho1 = torch::tensor({ phy.rho1 }, opts);
		rho2 = torch::tensor({ phy.rho2 }, opts);

		Y1 = torch::tensor({ phy.Y1 }, opts);
		Y2 = torch::tensor({ phy.Y2 }, opts);

		B1 = torch::tensor({ phy.B1 }, opts);
		B2 = torch::tensor({ phy.B2 }, opts);

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
		h = _h;
		is_para = _is_para;
		woven_pattern = WovenPattern(phy.woven_pattern);

		rho1_range = _rho1_range;
		rho2_range = _rho2_range;

		Y1_range = _Y1_range;
		Y2_range = _Y2_range;

		B1_range = _B1_range;
		B2_range = _B2_range;

		Tensor rho1_constraint = constraint_para(rho1, rho1_range.first, rho1_range.second);
		Tensor rho2_constraint = constraint_para(rho2, rho2_range.first, rho2_range.second);

		Tensor Y1_constraint = constraint_para(Y1, Y1_range.first, Y1_range.second);
		Tensor Y2_constraint = constraint_para(Y2, Y2_range.first, Y2_range.second);

		Tensor B1_constraint = constraint_para(B1, B1_range.first, B1_range.second);
		Tensor B2_constraint = constraint_para(B2, B2_range.first, B2_range.second);

		yarn1 = Yarn{ R, rho1_constraint, Y1_constraint, B1_constraint };
		yarn2 = Yarn{ R, rho2_constraint, Y2_constraint, B2_constraint };

		yarns = std::vector{ yarn1, yarn2 };
	}

	void reset_cloth() {
		Tensor rho1_constraint = constraint_para(rho1, rho1_range.first, rho1_range.second);
		Tensor rho2_constraint = constraint_para(rho2, rho2_range.first, rho2_range.second);

		Tensor Y1_constraint = constraint_para(Y1, Y1_range.first, Y1_range.second);
		Tensor Y2_constraint = constraint_para(Y2, Y2_range.first, Y2_range.second);

		Tensor B1_constraint = constraint_para(B1, B1_range.first, B1_range.second);
		Tensor B2_constraint = constraint_para(B2, B2_range.first, B2_range.second);

		yarns[0] = Yarn{ R, rho1_constraint, Y1_constraint, B1_constraint };
		yarns[1] = Yarn{ R, rho2_constraint, Y2_constraint, B2_constraint };

		cloth = Cloth(width, length, hight, L, R, yarns, kc, kf, df, mu, S, handle_stiff, woven_pattern, InitPose::Upright);

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

	int width, length;
	double hight, R, L;
	Tensor rho1, rho2;
	Tensor Y1, Y2;
	Tensor B1, B2;
	Tensor G, S, kf, df, mu, kc;
	Tensor handle_stiff;
	Wind wind;
	Yarn yarn1, yarn2;
	std::vector<Yarn> yarns;
	WovenPattern woven_pattern;
	Tensor h;
	bool is_para;
	std::pair<double, double> rho1_range, rho2_range;
	std::pair<double, double> Y1_range, Y2_range;
	std::pair<double, double> B1_range, B2_range;
	std::vector<std::pair<int, int>> handles;
	Cloth cloth;
	Environment env;
	PhysicsCloth phy_sim;
};
