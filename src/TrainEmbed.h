#pragma once

#include "ForceNet.h"
#include "YarnSimNet.h"
#include "Saver.h"
#include "Random.h"
#include "boost/format.hpp"
#include "indicators/progress_bar.hpp"

using namespace indicators;

void train_embed(std::string save_path, std::string data_path, int epoch, int iter, int batch, int sample, int steps, bool all_steps, bool random)
{
	// epoch: the number of training epoches
// iter: 
// 1.if not random, the number of bacthes in each iteration 
// (save in variable iteration) (see all training data)
// 2. if random, use the give iter as iteration.
// batch: the number of samples in each batch
// sample: the number of steps in each sample
// steps : the number of simulation steps in traing data

//torch::autograd::DetectAnomalyGuard detect_anomaly;

// file in which saves training result
	H5File file{ save_path, H5F_ACC_TRUNC };

	int samples{ steps / sample }; // the number of sample of training dataset
	int iteration{ 0 };
	if (all_steps)
		iteration = random ? iter : (samples / batch);
	else
		iteration = iter;
	int train_steps{ epoch * iteration }; // the number of training steps (backprop steps)

	RandInt SeleSample{ 0, samples - 1 }; // For Randomly Select Sample

	auto SizePhy = cloth_property_loader(data_path); // Read Cloth Size and Cloth Physical Properties

	H5ClothSize size{ SizePhy.first };
	H5Physical phy{ SizePhy.second };

	double height{ 5.0 };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(4, 0));
	handles.push_back(std::make_pair(0, 4));
	handles.push_back(std::make_pair(4, 4));

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	Tensor rho = torch::tensor({ phy.rho }, opts);
	Tensor G = torch::tensor({ 0.0, 0.0, phy.gravity }, opts).view({ 3,1 });
	Tensor Y = torch::tensor({ phy.Y }, opts);
	Tensor B = torch::tensor({ phy.B }, opts);
	Tensor S = torch::tensor({ phy.S }, opts);
	Tensor kf = torch::tensor({ phy.kf }, opts);
	Tensor df = torch::tensor({ phy.df }, opts);
	Tensor mu = torch::tensor({ phy.mu }, opts);
	Tensor kc = torch::tensor({ phy.kc }, opts);

	std::vector<Yarn> yarns;

	Yarn yarn{ phy.R , rho, Y, B };

	yarns.push_back(yarn);

	Tensor handle_stiff = torch::tensor({ phy.handle_stiffness }, opts);

	Tensor wind_vel = torch::tensor({ phy.wind_velocity_x, phy.wind_velocity_y ,phy.wind_velocity_z }, opts).view({ 3,1 });
	Tensor wind_density = torch::tensor({ phy.wind_density }, opts);
	Tensor wind_drag = torch::tensor({ phy.wind_drag }, opts);

	Wind wind{ wind_vel, wind_density, wind_drag };

	// Cloth Initialization

	// Net for Predicting Force

	int cloth_state_size = size.width * size.length * 3 + size.width * size.length * 2;	// Pos and Vel Vector Components Number

	int force_tensor_size = size.width * size.length * 3 + (size.width - 2) * (size.length - 2) * 2;	// Force Vector Components Number

	int net_intput_tensor_size = cloth_state_size * 2; // Pos and Vel Vector are concatenated and given to force_net

	int net_output_tensor_size = 2;	// lifting force strength and dragging force strength

	Tensor lift_force_position_tensor = torch::zeros({ force_tensor_size , 1 }, opts);

	Tensor drag_force_position_tensor = torch::zeros({ force_tensor_size , 1 }, opts);

	Tensor zero_ext_force = torch::zeros({ force_tensor_size , 1 }, opts);

	for (auto& hd : handles)
	{
		int lift_position = (hd.first * size.width + hd.second) * 3 + 2;
		int drag_position = (hd.first * size.width + hd.second) * 3 + 1;

		lift_force_position_tensor.index_put_({ lift_position, 0 }, 1.0);
		drag_force_position_tensor.index_put_({ drag_position, 0 }, 1.0);
	}

	ForceNet force_net{ net_intput_tensor_size,  net_output_tensor_size };

	std::vector<Tensor> lift_forces(6, torch::zeros({ 1 }, opts));
	std::vector<Tensor> drag_forces(6, torch::zeros({ 1 }, opts));

	int force_stop_frame = 5;

	ClothState clothStateEnd = data_loader(data_path, steps - 1, steps, opts); // Load cloth goal state

	ClothState clothStateGoal{
		clothStateEnd.LPos[0],
		clothStateEnd.LPosFix[0],
		clothStateEnd.LVel[0],
		clothStateEnd.EPos[0],
		clothStateEnd.EPosBar[0],
		clothStateEnd.EVel[0] };

	torch::optim::Adam optimizer(force_net->parameters(), 0.01);

	for (int e = 0; e < epoch; ++e)
	{
		std::cout << "Epoch: " << e << std::endl;

		Cloth cloth{ size.width, size.length, height, phy.L, phy.R, yarns, kc, kf, df, mu, S, handle_stiff, WovenPattern::Plain, InitPose::Upright };

		Environment env{ &cloth, G, wind };

		PhysicsCloth phy_sim{ &env };
		
		phy_sim.InitMV();

		bool is_success = true;

		ClothState bkupState = cloth.get_ClothState();

		for (int s = 0; s < sample - 1; ++s)
		{
			std::string objname = boost::str(boost::format("Cloth%|04|") % s);
			std::string save_path = "./debug_objs/" + objname + ".obj";

			if (s <= force_stop_frame)
			{

				Tensor cloth_state_l_pos = cloth.get_ClothState().LPos.flatten();
				Tensor cloth_state_e_pos = cloth.get_ClothState().EPos.flatten();
				Tensor cloth_state_l_vel = cloth.get_ClothState().LVel.flatten();
				Tensor cloth_state_e_vel = cloth.get_ClothState().EVel.flatten();

				Tensor cloth_state_input = torch::cat({ cloth_state_l_pos, cloth_state_e_pos, cloth_state_l_vel, cloth_state_e_vel }).unsqueeze(0).to(torch::kFloat32);

				Tensor pred_lift_drag_force = force_net->forward(cloth_state_input);

				Tensor pred_ext_force = lift_force_position_tensor * pred_lift_drag_force.index({ 0,0 }) + drag_force_position_tensor * pred_lift_drag_force.index({ 0,1 });

				lift_forces[s] = pred_lift_drag_force.index({ 0,0 });
				drag_forces[s] = pred_lift_drag_force.index({ 0,1 });

				is_success = SimStep(cloth, phy_sim, h, pred_ext_force);

				if (s = force_stop_frame)
				{
					bkupState = cloth.get_ClothState();
				}

				if (!is_success) 
				{
					break;
				}
			}
			else
			{
				is_success = SimStep(cloth, phy_sim, h, zero_ext_force);
				
				if (!is_success)
				{
					break;
				}
			}

			RenderCloth(cloth, save_path, RenderType::MeshTriangle);

		}

		ClothState finish_state = cloth.get_ClothState();

		if (!is_success)
			finish_state = bkupState;

		Tensor GT_LPos{ clothStateGoal.LPos }; // Ground Truth Lagrangian Position

		Tensor mass_center_distance{ (torch::mean(GT_LPos, 0) - torch::mean(finish_state.LPos, 0)).squeeze() };

		Tensor mean_distance_L{ torch::matmul(mass_center_distance, mass_center_distance) };

		Tensor loss = mean_distance_L;

		loss.backward();

		optimizer.step();

		std::cout << "Distance: " << loss << std::endl;

		std::map<std::string, Tensor> predicted_forces;

		predicted_forces.insert({ "lift_one", lift_forces[0] });
		predicted_forces.insert({ "drag_one", drag_forces[0] });

		predicted_forces.insert({ "lift_two", lift_forces[1] });
		predicted_forces.insert({ "drag_two", drag_forces[1] });

		predicted_forces.insert({ "lift_three", lift_forces[2] });
		predicted_forces.insert({ "drag_three", drag_forces[2] });

		predicted_forces.insert({ "lift_four", lift_forces[3] });
		predicted_forces.insert({ "drag_four", drag_forces[3] });

		predicted_forces.insert({ "lift_five", lift_forces[4] });
		predicted_forces.insert({ "drag_five", drag_forces[4] });

		predicted_forces.insert({ "lift_six", lift_forces[5] });
		predicted_forces.insert({ "drag_six", drag_forces[5] });

		SaveTrainProc(file, e, train_steps, loss, predicted_forces); // save loss and parameters

	}

	std::cout << "Finish Testing" << std::endl;

}
