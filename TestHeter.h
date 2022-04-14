#pragma once
#include <fstream>
#include <iomanip>
#include "YarnSimNetHeter.h"
#include "Saver.h"
#include "Random.h"
#include "indicators/progress_bar.hpp"
#include "boost/format.hpp"
#include <cmath>

using namespace indicators;

void test_heter(std::string save_path, std::string test_data_path, std::string objs_path, int sample, int steps) {

	/*
	1. save_path: file location where results to be saved
	2. test_data_path: file location of test data located
	*/
	
	torch::NoGradGuard no_grad;

	auto SizePhy = cloth_property_loader_heter(test_data_path); // Read Cloth Size and Cloth Physical Properties

	H5ClothSize size{ SizePhy.first };
	H5PhysicalHeterYarn phy{ SizePhy.second };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(16, 0));

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	double init_rho1{ 0.002156529962536863 };
	double init_rho2{ 0.002395332647888709 };

	double init_Y1{ 76040.39972129984 };
	double init_Y2{ 239726.29626741575 };

	double init_B1{ 1.7910772362198502E-4 }; 
	double init_B2{ 1.7808418890094763E-4 };

	double init_S{ 3264.366798661367 };

	auto rho1_range = std::make_pair(0.003, 0.001);
	auto rho2_range = std::make_pair(0.003, 0.001);

	auto Y1_range = std::make_pair(800000.0, 0.0);
	auto Y2_range = std::make_pair(300000.0, 0.0);

	auto B1_range = std::make_pair(0.00018, 0.00005);
	auto B2_range = std::make_pair(0.00018, 0.00005);

	auto S_range = std::make_pair(5000, 0);

	phy.rho1 = invert_constraint_para(init_rho1, rho1_range.first, rho1_range.second);
	phy.rho2 = invert_constraint_para(init_rho2, rho2_range.first, rho2_range.second);

	phy.Y1 = invert_constraint_para(init_Y1, Y1_range.first, Y1_range.second);
	phy.Y2 = invert_constraint_para(init_Y2, Y2_range.first, Y2_range.second);

	phy.B1 = invert_constraint_para(init_B1, B1_range.first, B1_range.second);
	phy.B2 = invert_constraint_para(init_B2, B2_range.first, B2_range.second);

	phy.S = invert_constraint_para(init_S, S_range.first, S_range.second);

	std::cout << "Testing Sample Size: " << std::to_string(sample - 1) << std::endl;

	YarnSimNetHeterMid simnet(size, phy, handles, h, rho1_range, rho2_range,
		Y1_range, Y2_range, B1_range, B2_range, S_range); // Construct Net Train

	int success_steps{ 0 }; // Success simulation steps
	Tensor loss{ torch::zeros(1, opts) };	// Create loss 

	simnet.reset_cloth();

	int sample_choice{ 0 };

	ClothState clothStateSample = data_loader(test_data_path, sample_choice * sample, sample_choice * sample + sample, opts); // Load training batch
	ClothState clothStateCurrt{
		clothStateSample.LPos[0],
		clothStateSample.LPosFix[0],
		clothStateSample.LVel[0],
		clothStateSample.EPos[0],
		clothStateSample.EPosBar[0],
		clothStateSample.EVel[0] }; // Cloth Initial State in this Batch. Later replaced by predicted cloth state

	ProgressBar bar{
		option::BarWidth{50},
		option::Start{"["},
		option::Fill{"="},
		option::Lead{">"},
		option::Remainder{" "},
		option::End{"]"},
		option::PostfixText{"Sample step 0"},
		option::ForegroundColor{Color::white},
		option::FontStyles{std::vector<FontStyle>{FontStyle::bold}} };
	float bar_step = 100.0f / (sample - 1);
	float bar_progress = 0.0f;

	for (int smp = 0; smp < sample - 1; ++smp) {
		std::string sample_step{ "Sample step: " + std::to_string(smp + 1) };

		std::string obj_name{ boost::str(boost::format("%|04|_00") % (smp + 1)) };
		std::string obj_full_path{ objs_path + obj_name + ".obj" };

		bool is_success{ false }; // Flag indicating if forward simulation success
		bar_progress += bar_step;
		bar.set_option(option::PostfixText{ sample_step });
		bar.set_progress(bar_progress);

		if (smp == 0)
			clothStateCurrt = simnet.forward(clothStateCurrt, is_success); // Initial sample: set cloth state
		else
			clothStateCurrt = simnet.forward(is_success); // Later samples: directly moving on

		if (!is_success) break; // Break out from the loop if NaN detected in simulating

		++success_steps;

		Tensor GT_LPos{ torch::zeros_like(clothStateCurrt.LPos) }; // Ground Truth Lagrangian Position
			
		load_obj_train(obj_full_path, GT_LPos);  // Load Cloth Lagrangian Position from obj file

		Tensor Dis_L{ clothStateCurrt.LPos - GT_LPos }; // Distance Lagrangian Position

		Tensor L2_Dis_L{ torch::bmm(Dis_L.transpose(2,1), Dis_L) }; // L2 Distance of nodes' Lagrangian Position

		Tensor PosLoss{ torch::mean(L2_Dis_L) }; // Mean of L2 Position distance at one step

		loss = loss + PosLoss;
	}

	loss = loss / ((success_steps - 1) > 0 ? (success_steps - 1) : 1);

	std::ofstream f;

	f.open(save_path);

	f << "Loss: " << std::setprecision(8) << loss.item<double>() << std::endl;

	f.close();

	std::cout << "Loss: " << std::setprecision(8) << loss.item<double>() << std::endl;

	std::cout << "Testing Done" << std::endl;
}

void test_heter_h5(std::string save_path, std::string data_path, int sample, int steps) {

	// epoch: the number of training epoches
	// iter: 
	// 1.if not random, the number of bacthes in each iteration 
	// (save in variable iteration) (see all training data)
	// 2. if random, use the give iter as iteration.
	// batch: the number of samples in each batch
	// sample: the number of steps in each sample
	// steps : the number of simulation steps in traing data

	//torch::autograd::DetectAnomalyGuard detect_anomaly;

	torch::NoGradGuard no_grad;

	auto SizePhy = cloth_property_loader_heter(data_path); // Read Cloth Size and Cloth Physical Properties

	H5ClothSize size{ SizePhy.first };
	H5PhysicalHeterYarn phy{ SizePhy.second };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(16, 0));

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	// Parameters for testing
	// To test the train model, replace the following lines by the code given in the 
	// 1. trained_parameters -> table 2
	// 2. trained_parameters -> appendix_table2
	// *** Begin: Code can be replaced ***
	double init_rho1{ 0.00225 };
	double init_rho2{ 0.00245 };

	double init_Y1{ 335000.0 };
	double init_Y2{ 145000.0 };

	double init_B1{ 0.000125 };
	double init_B2{ 0.0001 }; 

	//double init_rho1{ 0.002 };
	//double init_rho2{ 0.0025 };

	//double init_Y1{ 500000.0 };
	//double init_Y2{ 170000.0 };

	//double init_B1{ 0.00014 };
	//double init_B2{ 0.00011 }; 
	// *** End: Code can be replaced ***

	auto rho1_range = std::make_pair(0.003, 0.001);
	auto rho2_range = std::make_pair(0.003, 0.001);

	auto Y1_range = std::make_pair(800000.0, 0.0);
	auto Y2_range = std::make_pair(300000.0, 0.0);

	auto B1_range = std::make_pair(0.00018, 0.00005);
	auto B2_range = std::make_pair(0.00018, 0.00005);

	phy.rho1 = invert_constraint_para(init_rho1, rho1_range.first, rho1_range.second);
	phy.rho2 = invert_constraint_para(init_rho2, rho2_range.first, rho2_range.second);

	phy.Y1 = invert_constraint_para(init_Y1, Y1_range.first, Y1_range.second);
	phy.Y2 = invert_constraint_para(init_Y2, Y2_range.first, Y2_range.second);

	phy.B1 = invert_constraint_para(init_B1, B1_range.first, B1_range.second);
	phy.B2 = invert_constraint_para(init_B2, B2_range.first, B2_range.second);

	std::cout << "Sample Size: " << std::to_string(sample - 1) << std::endl;

	YarnSimNetHeterFew simnet(size, phy, handles, h, rho1_range, rho2_range,
		Y1_range, Y2_range, B1_range, B2_range); // Construct Net Train

	int success_steps{ 0 }; // Success simulation steps

	Tensor loss{ torch::zeros(1, opts) };	// Create loss 
	Tensor loss_no_uv{ torch::zeros(1, opts) }; // Loss function exclude uv coordinates for testing

	simnet.reset_cloth();

	int sample_choice{ 0 };

	ClothState clothStateSample = data_loader(data_path, sample_choice * sample, sample_choice * sample + sample, opts); // Load training batch
	ClothState clothStateCurrt{
		clothStateSample.LPos[0],
		clothStateSample.LPosFix[0],
		clothStateSample.LVel[0],
		clothStateSample.EPos[0],
		clothStateSample.EPosBar[0],
		clothStateSample.EVel[0] }; // Cloth Initial State in this Batch. Later replaced by predicted cloth state

	ProgressBar bar{
		option::BarWidth{50},
		option::Start{"["},
		option::Fill{"="},
		option::Lead{">"},
		option::Remainder{" "},
		option::End{"]"},
		option::PostfixText{"Sample step 0"},
		option::ForegroundColor{Color::white},
		option::FontStyles{std::vector<FontStyle>{FontStyle::bold}} };
	float bar_step = 100.0f / (sample - 1);
	float bar_progress = 0.0f;

	for (int smp = 0; smp < sample - 1; ++smp) {
		std::string sample_step{ "Sample step: " + std::to_string(smp + 1) };

		bool is_success{ false }; // Flag indicating if forward simulation success
		bar_progress += bar_step;
		bar.set_option(option::PostfixText{ sample_step });
		bar.set_progress(bar_progress);

		// (the reason for "smp == 0 && false")
		// Initial state can be directly set by cloth constructor, 
		// so I do not set cloth state from the read data. 
		if (smp == 0 && false)
			clothStateCurrt = simnet.forward(clothStateCurrt, is_success); // Initial sample: set cloth state
		else
			clothStateCurrt = simnet.forward(is_success); // Later samples: directly moving on

		if (!is_success) break; // Break out from the loop if NaN detected in simulating

		++success_steps;

		Tensor GT_LPos{ clothStateSample.LPos[smp + 1] }; // Ground Truth Lagrangian Position
		Tensor GT_EPos{ clothStateSample.EPos[smp + 1] }; // Ground Truth Euler Position

		Tensor Dis_L{ clothStateCurrt.LPos - GT_LPos }; // Distance Lagrangian Position
		Tensor Dis_E{ clothStateCurrt.EPos - GT_EPos }; // Distance Euler Position

		Tensor L2_Dis_L{ torch::bmm(Dis_L.transpose(2,1), Dis_L) }; // L2 Distance of nodes' Lagrangian Position
		Tensor L2_Dis_E{ torch::bmm(Dis_E.transpose(2,1), Dis_E) }; // L2 Distance of nodes' Euler Position

		Tensor PosLoss{ torch::mean(L2_Dis_L) + torch::mean(L2_Dis_E) }; // Mean of L2 Position distance at one step

		Tensor PosLoss_no_uv{ torch::mean(L2_Dis_L) }; // Mean of L2 Position distance at one step

		loss = loss + PosLoss;
		loss_no_uv = loss_no_uv + PosLoss_no_uv;
	}

	loss = loss / ((success_steps - 1) > 0 ? (success_steps - 1) : 1); // Mean of loss of one sample (sample-1 losses)		
	loss_no_uv = loss_no_uv / ((success_steps - 1) > 0 ? (success_steps - 1) : 1);

	std::cout << "Loss: " << loss.item<double>() << std::endl;
	std::cout << "Loss no uv: " << loss_no_uv.item<double>() << std::endl;
}


