#pragma once
#include "Render.h"
#include "TrainHelper.h"

struct YarnSimNet : torch::nn::Module {

	YarnSimNet(H5ClothSize size, H5Physical phy, std::vector<std::pair<int, int>> _handles, Tensor _h, std::pair<double, double> _rho_range, std::pair<double, double> _Y_range, std::pair<double, double> _B_range):
		width(size.width), length(size.length), height(5.0), R(phy.R), L(phy.L){	
		G = torch::tensor({ 0.0, 0.0, phy.gravity }, opts).view({ 3,1 });

		rho = register_parameter("rho", torch::tensor({ phy.rho }, opts));
		Y = register_parameter("Y", torch::tensor({ phy.Y }, opts));
		B = register_parameter("B", torch::tensor({ phy.B }, opts));

		S = torch::tensor({ phy.S }, opts);
		kf = torch::tensor({ phy.kf }, opts);
		df = torch::tensor({ phy.df }, opts);
		mu = torch::tensor({ phy.mu }, opts);
		kc = torch::tensor({ phy.kc }, opts);

		handle_stiff = torch::tensor({phy.handle_stiffness}, opts);

		Tensor wind_vel = torch::tensor({ phy.wind_velocity_x, phy.wind_velocity_y ,phy.wind_velocity_z }, opts).view({ 3,1 });
		Tensor wind_density = torch::tensor({ phy.wind_density }, opts);
		Tensor wind_drag = torch::tensor({ phy.wind_drag }, opts);

		wind = Wind(wind_vel, wind_density, wind_drag);

		h = _h;

		rho_range = _rho_range;
		Y_range = _Y_range;
		B_range = _B_range;

		handles = _handles;

		Yarn yarn{R, rho, Y, B};

		yarns = std::vector{ yarn };
	}

	void reset_cloth() {

		Tensor rho_constrained = constraint_para(rho, rho_range.first, rho_range.second);
		Tensor Y_constrained = constraint_para(Y, Y_range.first, Y_range.second);
		Tensor B_constrained = constraint_para(B, B_range.first, B_range.second);

		yarns[0] = Yarn(R, rho_constrained, Y_constrained, B_constrained);

		cloth = Cloth(width, length, height, L, R, yarns, kc, kf, df, mu, S, handle_stiff, WovenPattern::Plain, InitPose::Upright);

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
	double height, R, L;
	Tensor rho, G, Y, B, S, kf, df, mu, kc;
	Tensor handle_stiff;
	Wind wind;
	Tensor h;
	std::pair<double, double> rho_range;
	std::pair<double, double> Y_range;
	std::pair<double, double> B_range;
	std::vector<std::pair<int, int>> handles;
	std::vector<Yarn> yarns;
	Cloth cloth;
	Environment env;
	PhysicsCloth phy_sim;
};

struct YarnSimNetControl : torch::nn::Module {
	YarnSimNetControl(H5ClothSize size, H5Physical phy,
		double _lift_force_one, double _drag_force_one,
		double _lift_force_two, double _drag_force_two,
		double _lift_force_three, double _drag_force_three,
		double _lift_force_four, double _drag_force_four,
		double _lift_force_five, double _drag_force_five,
		double _lift_force_six, double _drag_force_six,
		std::vector<std::pair<int, int>> _handles, Tensor _h) :
		width(size.width), length(size.length), height(5.0), R(phy.R), L(phy.L) {

		lift_force_one = register_parameter("lift_force_one", torch::tensor({ _lift_force_one }, opts));
		drag_force_one = register_parameter("drag_force_one", torch::tensor({ _drag_force_one }, opts));

		lift_force_two = register_parameter("lift_force_two", torch::tensor({ _lift_force_two }, opts));
		drag_force_two = register_parameter("drag_force_two", torch::tensor({ _drag_force_two }, opts));

		lift_force_three = register_parameter("lift_force_three", torch::tensor({ _lift_force_three }, opts));
		drag_force_three = register_parameter("drag_force_three", torch::tensor({ _drag_force_three }, opts));

		lift_force_four = register_parameter("lift_force_four", torch::tensor({ _lift_force_four }, opts));
		drag_force_four = register_parameter("drag_force_four", torch::tensor({ _drag_force_four }, opts));

		lift_force_five = register_parameter("lift_force_five", torch::tensor({ _lift_force_five }, opts));
		drag_force_five = register_parameter("drag_force_five", torch::tensor({ _drag_force_five }, opts));

		lift_force_six = register_parameter("lift_force_six", torch::tensor({ _lift_force_six }, opts));
		drag_force_six = register_parameter("drag_force_six", torch::tensor({ _drag_force_six }, opts));

		rho = torch::tensor({ phy.rho }, opts);
		G = torch::tensor({ 0.0, 0.0, phy.gravity }, opts).view({3,1});
		Y = torch::tensor({ phy.Y }, opts);
		B = torch::tensor({ phy.B }, opts);
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

		force_tensor_size = width * length * 3 + (width - 2) * (length - 2) * 2;

		lift_force_position_tensor = torch::zeros({ force_tensor_size , 1 }, opts);
		drag_force_position_tensor = torch::zeros({ force_tensor_size , 1 }, opts);

		zero_ext_force = torch::zeros({ force_tensor_size , 1 }, opts);

		for (auto& h : handles)
		{
			int lift_position = (h.first * width + h.second) * 3 + 2;
			int drag_position = (h.first * width + h.second) * 3 + 1;

			lift_force_position_tensor.index_put_({ lift_position, 0 }, 1.0);
			drag_force_position_tensor.index_put_({ drag_position, 0 }, 1.0);
		}

		h = _h;

		Yarn yarn{ R, rho, Y, B };

		yarns = std::vector{ yarn };

	}

	void reset_cloth() {

		ext_force_one = lift_force_position_tensor * lift_force_one + drag_force_position_tensor * drag_force_one;
		ext_force_two = lift_force_position_tensor * lift_force_two + drag_force_position_tensor * drag_force_two;
		ext_force_three = lift_force_position_tensor * lift_force_three + drag_force_position_tensor * drag_force_three;
		ext_force_four = lift_force_position_tensor * lift_force_four + drag_force_position_tensor * drag_force_four;
		ext_force_five = lift_force_position_tensor * lift_force_five + drag_force_position_tensor * drag_force_five;
		ext_force_six = lift_force_position_tensor * lift_force_six + drag_force_position_tensor * drag_force_six;

		cloth = Cloth(width, length, height, L, R, yarns, kc, kf, df, mu, S, handle_stiff, WovenPattern::Plain, InitPose::Upright);

		env = Environment(&cloth, G, wind);

		phy_sim = PhysicsCloth(&env);
		phy_sim.InitMV();
	}

	ClothState forward(ClothState cloth_state, bool& is_success, int ext_force_idx) {
		cloth.set_ClothState(cloth_state);

		if (ext_force_idx == 0)
			is_success = SimStep(cloth, phy_sim, h, ext_force_one);
		else if (ext_force_idx == 1)
			is_success = SimStep(cloth, phy_sim, h, ext_force_two);
		else if (ext_force_idx == 2)
			is_success = SimStep(cloth, phy_sim, h, ext_force_three);
		else if (ext_force_idx == 3)
			is_success = SimStep(cloth, phy_sim, h, ext_force_four);
		else if (ext_force_idx == 4)
			is_success = SimStep(cloth, phy_sim, h, ext_force_five);
		else if (ext_force_idx == 5)
			is_success = SimStep(cloth, phy_sim, h, ext_force_six);
		else if (ext_force_idx == 6)
			is_success = SimStep(cloth, phy_sim, h, zero_ext_force);
		else
			std::cout << "Invalid force index" << std::endl;
			
		return cloth.get_ClothState();
	}

	ClothState forward(bool& is_success, std::string save_path, int ext_force_idx) {

		if (ext_force_idx == 0)
			is_success = SimStep(cloth, phy_sim, h, ext_force_one);
		else if (ext_force_idx == 1)
			is_success = SimStep(cloth, phy_sim, h, ext_force_two);
		else if (ext_force_idx == 2)
			is_success = SimStep(cloth, phy_sim, h, ext_force_three);
		else if (ext_force_idx == 3)
			is_success = SimStep(cloth, phy_sim, h, ext_force_four);
		else if (ext_force_idx == 4)
			is_success = SimStep(cloth, phy_sim, h, ext_force_five);
		else if (ext_force_idx == 5)
			is_success = SimStep(cloth, phy_sim, h, ext_force_six);
		else if (ext_force_idx == 6)
			is_success = SimStep(cloth, phy_sim, h, zero_ext_force);
		else
			std::cout << "Invalid force index" << std::endl;

		RenderCloth(cloth, save_path, RenderType::MeshTriangle);

		return cloth.get_ClothState();
	}

	int width, length;
	double height, R, L;

	Tensor lift_force_one, drag_force_one;
	Tensor lift_force_two, drag_force_two;
	Tensor lift_force_three, drag_force_three;
	Tensor lift_force_four, drag_force_four;
	Tensor lift_force_five, drag_force_five;
	Tensor lift_force_six, drag_force_six;

	Tensor rho, G, Y, B, S, kf, df, mu, kc;
	Tensor handle_stiff;
	Wind wind;
	Tensor h;
	std::vector<std::pair<int, int>> handles;
	int force_tensor_size;
	Tensor lift_force_position_tensor, drag_force_position_tensor;
	Tensor ext_force_one, ext_force_two, ext_force_three, ext_force_four, ext_force_five, ext_force_six, zero_ext_force;
	std::vector<Yarn> yarns;
	Cloth cloth;
	Environment env;
	PhysicsCloth phy_sim;
};