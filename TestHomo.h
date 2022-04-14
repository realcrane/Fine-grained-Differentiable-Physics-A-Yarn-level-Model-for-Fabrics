#pragma once
#pragma once
#include <fstream>
#include <iomanip>
#include "YarnSimNetHomo.h"
#include "Saver.h"
#include "Random.h"
#include "indicators/progress_bar.hpp"
#include "boost/format.hpp"
#include <cmath>

using namespace indicators;

void test_homo(std::string save_path, std::string test_data_path, std::string objs_path, int sample, int steps) {

	/*
	1. save_path: file location where results to be saved
	2. test_data_path: file location of test data located
	*/

	torch::NoGradGuard no_grad;

	auto SizePhy = cloth_property_loader(test_data_path); // Read Cloth Size and Cloth Physical Properties

	H5ClothSize size{ SizePhy.first };
	H5Physical phy{ SizePhy.second };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(16, 0));

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	// Parameters for testing
	// To test the train model, replace the following lines by the code given in the 
	// 1. trained_parameters -> table 2
	// 2. trained_parameters -> appendix_table2
	// *** Begin: Code can be replaced ***
	double init_rho{ 0.002607734360971621 };
	double init_Y{ 1581836.2112229776 };
	double init_B{ 1.7706782538366917E-4 };
	double init_S{ 27725.223551164614 };
	// *** End: Code can be replaced ***

	auto rho_range = std::make_pair(0.008, 0.001);
	auto Y_range = std::make_pair(2000000.0, 0.0);
	auto B_range = std::make_pair(0.0002, 0.0);
	auto S_range = std::make_pair(30000, 0);

	phy.rho = invert_constraint_para(init_rho, rho_range.first, rho_range.second);
	phy.Y = invert_constraint_para(init_Y, Y_range.first, Y_range.second);
	phy.B = invert_constraint_para(init_B, B_range.first, B_range.second);
	phy.S = invert_constraint_para(init_S, S_range.first, S_range.second);

	std::cout << "Testing Sample Size: " << std::to_string(sample - 1) << std::endl;

	YarnSimNetHomo simnet(size, phy, handles, h, rho_range, Y_range, B_range, S_range); // Construct Net Train

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

