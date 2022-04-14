#pragma once
#include "YarnSimNetHeter.h"
#include "Saver.h"
#include "Random.h"
#include "indicators/progress_bar.hpp"
#include "boost/format.hpp"
#include <cmath>

using namespace indicators;

void train_heter_full(std::string save_path, std::string data_path, 
	int epoch, int iter, int batch, int sample, int steps, 
	bool all_steps, bool random, bool wordy) {

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
	int iteration{ 0 }; // the number of training iteration
	if (all_steps)
		iteration = random ? iter : (samples / batch);
	else
		iteration = iter;
	int train_steps{ epoch * iteration }; // the number of training steps (backprop steps)

	RandInt SeleSample{ 0, samples - 1 }; // For Randomly Select Sample

	auto SizePhy = cloth_property_loader_heter(data_path); // Read Cloth Size and Cloth Physical Properties

	H5ClothSize size{ SizePhy.first };
	H5PhysicalHeterYarn phy{ SizePhy.second };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(16, 0));  // Alter this handle when changing cloth size

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	// Initialization Parameters
	// Parameters for training
	double init_rho1{ 0.00225 };
	double init_rho2{ 0.00245 };

	double init_Y1{ 335000.0 };
	double init_Y2{ 145000.0 };

	double init_B1{ 0.000125 };
	double init_B2{ 0.0001 };

	double init_S{ 600 };

	double init_mu{0.7};

	// Gt Parameters
	//double init_rho1{ 0.002 };
	//double init_rho2{ 0.0025 };

	//double init_Y1{ 500000.0 };
	//double init_Y2{ 170000.0 };

	//double init_B1{ 0.00014 };
	//double init_B2{ 0.00011 };

	//double init_S{ 1000 };

	//double init_mu{ 0.5 };
 
	// Parameters' range
	auto rho1_range = std::make_pair(0.003, 0.001);
	auto rho2_range = std::make_pair(0.003, 0.001);

	auto Y1_range = std::make_pair(800000.0, 0.0);
	auto Y2_range = std::make_pair(300000.0, 0.0);

	auto B1_range = std::make_pair(0.00018, 0.00005);
	auto B2_range = std::make_pair(0.00018, 0.00005);

	auto S_range = std::make_pair(1200.0, 0.0);

	auto mu_range = std::make_pair(1.0, 0.0);

	phy.rho1 = invert_constraint_para(init_rho1, rho1_range.first, rho1_range.second);
	phy.rho2 = invert_constraint_para(init_rho2, rho2_range.first, rho2_range.second);

	phy.Y1 = invert_constraint_para(init_Y1, Y1_range.first, Y1_range.second);
	phy.Y2 = invert_constraint_para(init_Y2, Y2_range.first, Y2_range.second);

	phy.B1 = invert_constraint_para(init_B1, B1_range.first, B1_range.second);
	phy.B2 = invert_constraint_para(init_B2, B2_range.first, B2_range.second);

	phy.S = invert_constraint_para(init_S, S_range.first, S_range.second);

	phy.mu = invert_constraint_para(init_mu, mu_range.first, mu_range.second);

	//phy.rho1 = init_rho1;
	//phy.rho2 = init_rho2;

	//phy.Y1 = init_Y1;
	//phy.Y2 = init_Y2;

	//phy.B1 = init_B1;
	//phy.B2 = init_B2;

	//phy.S = init_S;

	//phy.mu = init_mu;

	std::cout << "Epoch: " << std::to_string(epoch) << std::endl;
	std::cout << "Iteration: " << std::to_string(iteration) << std::endl;
	std::cout << "Batch Size: " << std::to_string(batch) << std::endl;
	std::cout << "Sample Size: " << std::to_string(sample - 1) << std::endl;
	std::cout << "Random Sampling: " << (random ? "True" : "False") << std::endl;

	YarnSimNetHeter simnet(size, phy, handles, h, rho1_range, rho2_range, Y1_range, Y2_range, B1_range, B2_range, S_range, mu_range); // Construct Net

	torch::optim::OptimizerParamGroup rho1({ simnet.named_parameters()["rho_1"] });
	torch::optim::OptimizerParamGroup rho2({ simnet.named_parameters()["rho_2"] });
	torch::optim::OptimizerParamGroup Y1({ simnet.named_parameters()["Y_1"] });
	torch::optim::OptimizerParamGroup Y2({ simnet.named_parameters()["Y_2"] });
	torch::optim::OptimizerParamGroup B1({ simnet.named_parameters()["B_1"] });
	torch::optim::OptimizerParamGroup B2({ simnet.named_parameters()["B_2"] });
	torch::optim::OptimizerParamGroup S({ simnet.named_parameters()["S"] });
	torch::optim::OptimizerParamGroup mu({ simnet.named_parameters()["mu"] });

	torch::optim::SGD cloth_opt({ rho1, rho2, Y1, Y2, B1, B2, S, mu}, torch::optim::SGDOptions(1e5));

	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[0].options()).lr(0.1); // lr rho1
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[1].options()).lr(1e-4); // lr rho2
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[2].options()).lr(12e16); // lr Y1
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[3].options()).lr(7e16); // lr Y2
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[4].options()).lr(0.219); // lr B1
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[5].options()).lr(0.145); // lr B2
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[6].options()).lr(19e10); // lr S
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[7].options()).lr(14e9); // lr mu


	for (int epc = 0; epc < epoch; ++epc) {
		std::cout << "Epoch: " << epc << std::endl;
		for (int i = 0; i < iteration; ++i) {
			// Update parameters each batch
			std::cout << "Batch: " << i << std::endl;
			int success_steps{ 0 }; // Success simulation steps
			int curr_step{ epc * iteration + i }; // cuttent step in all steps
			Tensor loss{ torch::zeros(1, opts) };	// Create loss 

			//Tensor loss_no_uv { torch::zeros(1, opts) }; // Loss function exclude uv coordinates for testing

			simnet.reset_cloth();
			cloth_opt.zero_grad();

			for (int bch = 0; bch < batch; ++bch) {
				std::cout << "Sample: " << std::to_string(bch + 1) << std::endl;

				int sample_choice{ 0 };
				if (random)
					sample_choice = SeleSample(); // Generate a random number
				else
					sample_choice = i * batch + bch; // Sequential retrieve

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

					if (smp == 0 || all_steps)
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

					loss = loss + PosLoss;
					//loss_no_uv = loss_no_uv + PosLoss_no_uv;
				}
			}
			loss = loss / ((success_steps - 1) > 0 ? (success_steps - 1) : 1); // Mean of loss of one sample (sample-1 losses)		
			//loss_no_uv = loss_no_uv / ((success_steps - 1) > 0 ? (success_steps - 1) : 1);
			//std::cout << "Loss: " << loss.item<double>() << std::endl;
			//std::cout << "Loss no uv: " << loss_no_uv.item<double>() << std::endl;

			loss.backward(); // compute grad of register paremeters

			if (wordy) {
				std::cout << "\nrho1 grad" << simnet.rho1.grad() << std::endl;
				std::cout << "rho2 grad" << simnet.rho2.grad() << std::endl;

				std::cout << "Y1 grad" << simnet.Y1.grad() << std::endl;
				std::cout << "Y2 grad" << simnet.Y2.grad() << std::endl;

				std::cout << "B1 grad" << simnet.B1.grad() << std::endl;
				std::cout << "B2 grad" << simnet.B2.grad() << std::endl;

				std::cout << "S grad" << simnet.S.grad() << std::endl;
				std::cout << "mu grad" << simnet.mu.grad() << std::endl;
			}

			bool valid_step{ true };

			for (const auto& para : simnet.parameters())
				if (torch::isnan(para.grad()).item<bool>() ||
					torch::isinf(para.grad()).item<bool>()) {
					valid_step = false;
					break;
				}

			if (valid_step)
				cloth_opt.step(); // Excute Optim

			auto rho1_constraint = constraint_para(simnet.rho1, rho1_range.first, rho1_range.second);
			auto rho2_constraint = constraint_para(simnet.rho2, rho2_range.first, rho2_range.second);

			auto Y1_constraint = constraint_para(simnet.Y1, Y1_range.first, Y1_range.second);
			auto Y2_constraint = constraint_para(simnet.Y2, Y2_range.first, Y2_range.second);

			auto B1_constraint = constraint_para(simnet.B1, B1_range.first, B1_range.second);
			auto B2_constraint = constraint_para(simnet.B2, B2_range.first, B2_range.second);

			auto S_constraint = constraint_para(simnet.S, S_range.first, S_range.second);

			auto mu_constraint = constraint_para(simnet.mu, mu_range.first, mu_range.second);

			if (wordy) {
				std::cout << "\nOptimized rho1: " << rho1_constraint.item<double>() << std::endl;
				std::cout << "Optimized rho2: " << rho2_constraint.item<double>() << std::endl;

				std::cout << "Optimized Y1: " << Y1_constraint.item<double>() << std::endl;
				std::cout << "Optimized Y2: " << Y2_constraint.item<double>() << std::endl;

				std::cout << "Optimized B1: " << B1_constraint.item<double>() << std::endl;
				std::cout << "Optimized B2: " << B2_constraint.item<double>() << std::endl;

				std::cout << "Optimized S: " << S_constraint.item<double>() << std::endl;
				std::cout << "Optimized mu: " << mu_constraint.item<double>() << std::endl;
			}

			std::cout << "Loss: " << loss.item<double>() << std::endl;

			std::map<std::string, Tensor> constrained_para;

			constrained_para.insert({ "rho1", rho1_constraint });
			constrained_para.insert({ "rho2", rho2_constraint });

			constrained_para.insert({ "Y1", Y1_constraint });
			constrained_para.insert({ "Y2", Y2_constraint });

			constrained_para.insert({ "B1", B1_constraint });
			constrained_para.insert({ "B2", B2_constraint });

			constrained_para.insert({ "S", S_constraint });
			constrained_para.insert({ "mu", mu_constraint });

			SaveTrainProc(file, curr_step, train_steps, loss, constrained_para); // save loss and parameters
		}
	}
	file.close();
}

void train_heter_few(std::string save_path, std::string data_path, 
	int epoch, int iter, int batch, int sample, int steps, 
	bool all_steps, bool random, bool wordy) {

	// epoch: the number of training epoches
	// iter: 
	// 1.if not random, the number of bacthes in each iteration 
	// (save in variable iteration) (see all training data)
	// 2. if random, use the give iter as iteration.
	// batch: the number of samples in each batch
	// sample: the number of steps in each sample
	// steps : the number of simulation steps in traing data

	// file in which saves training result
	H5File file{ save_path, H5F_ACC_TRUNC };

	int samples{ steps / sample }; // the number of sample of training dataset
	int iteration{ 0 }; // the number of training iteration
	if (all_steps)
		iteration = random ? iter : (samples / batch);
	else
		iteration = iter;
	int train_steps{ epoch * iteration }; // the number of training steps (backprop steps)

	RandInt SeleSample{ 0, samples - 1 }; // For Randomly Select Sample

	auto SizePhy = cloth_property_loader_heter(data_path); // Read Cloth Size and Cloth Physical Properties

	H5ClothSize size{ SizePhy.first };
	H5PhysicalHeterYarn phy{ SizePhy.second };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));

	// *** Change Handle below when changing cloth size ***
	// Cloth Size 5 x 5: std::make_pair(4, 0)
	// Cloth Size 10 x 10: std::make_pair(9, 0)
	// Cloth Size 17 x 17: std::make_pair(16, 0)
	// Cloth Size 17 x 17: std::make_pair(24, 0)
	handles.push_back(std::make_pair(4, 0));  

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	// Initialization Parameters
	// Parameters for training
	double init_rho1{ 0.00225 }; // 0.002
	double init_rho2{ 0.00245 }; // 0.0025

	double init_Y1{ 335000.0 }; // 500000 
	double init_Y2{ 145000.0 }; // 170000 

	double init_B1{ 0.000125 }; //  0.00014 
	double init_B2{ 0.0001 }; //  0.00011 

	// GTs
	//double init_rho1{ phy.rho1 }; // 0.002
	//double init_rho2{ phy.rho2 }; // 0.0025

	//double init_Y1{ phy.Y1 }; // 500000
	//double init_Y2{ phy.Y2 }; // 170000

	//double init_B1{ phy.B1 }; //  0.00014 
	//double init_B2{ phy.B2 }; //  0.00011

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

	std::cout << "Epoch: " << std::to_string(epoch) << std::endl;
	std::cout << "Iteration: " << std::to_string(iteration) << std::endl;
	std::cout << "Batch Size: " << std::to_string(batch) << std::endl;
	std::cout << "Sample Size: " << std::to_string(sample - 1) << std::endl;
	std::cout << "Random Sampling: " << (random ? "True" : "False") << std::endl;

	YarnSimNetHeterFew simnet(size, phy, handles, h, rho1_range, rho2_range,
		Y1_range, Y2_range, B1_range, B2_range); // Construct Net Train

	torch::optim::OptimizerParamGroup rho1({ simnet.named_parameters()["rho_1"] });
	torch::optim::OptimizerParamGroup rho2({ simnet.named_parameters()["rho_2"] });
	torch::optim::OptimizerParamGroup Y1({ simnet.named_parameters()["Y_1"] });
	torch::optim::OptimizerParamGroup Y2({ simnet.named_parameters()["Y_2"] });
	torch::optim::OptimizerParamGroup B1({ simnet.named_parameters()["B_1"] });
	torch::optim::OptimizerParamGroup B2({ simnet.named_parameters()["B_2"] });

	torch::optim::SGD cloth_opt({ rho1, rho2, Y1, Y2, B1, B2 }, torch::optim::SGDOptions(1e5));

	// ***Altering Parameters' learning rate here***
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[0].options()).lr(38e5); // lr rho1
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[1].options()).lr(13e5); // lr rho2
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[2].options()).lr(11e6); // lr Y1
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[3].options()).lr(18e6); // lr Y2
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[4].options()).lr(5e8); // lr B1
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[5].options()).lr(2e8); // lr B2

	for (int epc = 0; epc < epoch; ++epc) {
		std::cout << "Epoch: " << epc << std::endl;
		for (int i = 0; i < iteration; ++i) {
			// Update parameters each batch
			std::cout << "Batch: " << i << std::endl;
			int success_steps{ 0 }; // Success simulation steps
			int curr_step{ epc * iteration + i }; // cuttent step in all steps
			Tensor loss{ torch::zeros(1, opts) };	// Create loss 

			simnet.reset_cloth();
			cloth_opt.zero_grad();

			for (int bch = 0; bch < batch; ++bch) {
				std::cout << "Sample: " << std::to_string(bch + 1) << std::endl;

				int sample_choice{ 0 };
				if (random)
					sample_choice = SeleSample(); // Generate a random number
				else
					sample_choice = i * batch + bch; // Sequential retrieve

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

					if (smp == 0 || all_steps)
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

					Tensor PosLoss_no_uv{ torch::mean(L2_Dis_L)}; // Mean of L2 Position distance at one step

					loss = loss + PosLoss;
				}
			}
			loss = loss / ((success_steps - 1) > 0 ? (success_steps - 1) : 1); // Mean of loss of one sample (sample-1 losses)		
			//loss_no_uv = loss_no_uv / ((success_steps - 1) > 0 ? (success_steps - 1) : 1);
			//std::cout << "Loss: " << loss.item<double>() << std::endl;
			//std::cout << "Loss no uv: " << loss_no_uv.item<double>() << std::endl;

			loss.backward(); // compute grad of register paremeters

			if (wordy) {
				std::cout << "\nrho1 grad" << simnet.rho1.grad() << std::endl;
				std::cout << "rho2 grad" << simnet.rho2.grad() << std::endl;

				std::cout << "Y1 grad" << simnet.Y1.grad() << std::endl;
				std::cout << "Y2 grad" << simnet.Y2.grad() << std::endl;

				std::cout << "B1 grad" << simnet.B1.grad() << std::endl;
				std::cout << "B2 grad" << simnet.B2.grad() << std::endl;
			}

			bool valid_step{ true };

			for (const auto& para : simnet.parameters())
				if (torch::isnan(para.grad()).item<bool>() ||
					torch::isinf(para.grad()).item<bool>()) {
					valid_step = false;
					break;
				}

			if (valid_step)
				cloth_opt.step(); // Excute Optim

			auto rho1_constraint = constraint_para(simnet.rho1, rho1_range.first, rho1_range.second);
			auto rho2_constraint = constraint_para(simnet.rho2, rho2_range.first, rho2_range.second);

			auto Y1_constraint = constraint_para(simnet.Y1, Y1_range.first, Y1_range.second);
			auto Y2_constraint = constraint_para(simnet.Y2, Y2_range.first, Y2_range.second);

			auto B1_constraint = constraint_para(simnet.B1, B1_range.first, B1_range.second);
			auto B2_constraint = constraint_para(simnet.B2, B2_range.first, B2_range.second);

			if (wordy) {
				std::cout << "\nOptimized rho1: " << rho1_constraint.item<double>() << std::endl;
				std::cout << "Optimized rho2: " << rho2_constraint.item<double>() << std::endl;

				std::cout << "Optimized Y1: " << Y1_constraint.item<double>() << std::endl;
				std::cout << "Optimized Y2: " << Y2_constraint.item<double>() << std::endl;

				std::cout << "Optimized B1: " << B1_constraint.item<double>() << std::endl;
				std::cout << "Optimized B2: " << B2_constraint.item<double>() << std::endl;
			}

			std::cout << "Loss: " << loss.item<double>() << std::endl;

			std::map<std::string, Tensor> constrained_para;

			constrained_para.insert({ "rho1", rho1_constraint });
			constrained_para.insert({ "rho2", rho2_constraint });

			constrained_para.insert({ "Y1", Y1_constraint });
			constrained_para.insert({ "Y2", Y2_constraint });

			constrained_para.insert({ "B1", B1_constraint });
			constrained_para.insert({ "B2", B2_constraint });

			SaveTrainProc(file, curr_step, train_steps, loss, constrained_para); // save loss and parameters
		}
	}
	file.close();
}

void train_heter_few_obj(std::string save_path, std::string data_path, std::string objs_path,
	int epoch, int iter, int batch, int sample, int steps, bool all_steps, bool random, bool para) {

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
	int iteration{ 0 }; // the number of training iteration
	if (all_steps)
		iteration = random ? iter : (samples / batch);
	else
		iteration = iter;
	int train_steps{ epoch * iteration }; // the number of training steps (backprop steps)

	RandInt SeleSample{ 0, samples - 1 }; // For Randomly Select Sample

	auto SizePhy = cloth_property_loader_heter(data_path); // Read Cloth Size and Cloth Physical Properties

	H5ClothSize size{ SizePhy.first };
	H5PhysicalHeterYarn phy{ SizePhy.second };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(16, 0));

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	// Initialization Parameters

	double init_rho1{ 0.00225 }; 
	double init_rho2{ 0.00245 }; 

	double init_Y1{ 335000.0 }; 
	double init_Y2{ 170000.0 }; 

	double init_B1{ 0.000125 }; 
	double init_B2{ 0.0001 }; 

	double init_S{ 1000 };

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

	std::cout << "Epoch: " << std::to_string(epoch) << std::endl;
	std::cout << "Iteration: " << std::to_string(iteration) << std::endl;
	std::cout << "Batch Size: " << std::to_string(batch) << std::endl;
	std::cout << "Sample Size: " << std::to_string(sample - 1) << std::endl;
	std::cout << "Random Sampling: " << (random ? "True" : "False") << std::endl;
	std::cout << "Parallel: " << (para ? "True" : "False") << std::endl;

	YarnSimNetHeterMid simnet(size, phy, handles, h, rho1_range, rho2_range,
		Y1_range, Y2_range, B1_range, B2_range, S_range); // Construct Net Train

	torch::optim::OptimizerParamGroup rho1({ simnet.named_parameters()["rho_1"] });
	torch::optim::OptimizerParamGroup rho2({ simnet.named_parameters()["rho_2"] });
	torch::optim::OptimizerParamGroup Y1({ simnet.named_parameters()["Y_1"] });
	torch::optim::OptimizerParamGroup Y2({ simnet.named_parameters()["Y_2"] });
	torch::optim::OptimizerParamGroup B1({ simnet.named_parameters()["B_1"] });
	torch::optim::OptimizerParamGroup B2({ simnet.named_parameters()["B_2"] });
	torch::optim::OptimizerParamGroup S({ simnet.named_parameters()["S"] });

	torch::optim::SGD cloth_opt({ rho1, rho2, Y1, Y2, B1, B2, S }, torch::optim::SGDOptions(1e5));

	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[0].options()).lr(4e6);
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[1].options()).lr(6e7);
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[2].options()).lr(1e7);
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[3].options()).lr(40e8);
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[4].options()).lr(6e9);
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[5].options()).lr(8e8);
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[6].options()).lr(3e7);

	for (int epc = 0; epc < epoch; ++epc) {
		std::cout << "Epoch: " << epc << std::endl;
		for (int i = 0; i < iteration; ++i) {
			// Update parameters each batch
			std::cout << "Batch: " << i << std::endl;
			int success_steps{ 0 }; // Success simulation steps
			int curr_step{ epc * iteration + i }; // cuttent step in all steps
			Tensor loss{ torch::zeros(1, opts) };	// Create loss 

			simnet.reset_cloth();
			cloth_opt.zero_grad();

			for (int bch = 0; bch < batch; ++bch) {
				std::cout << "Sample: " << std::to_string(bch + 1) << std::endl;

				int sample_choice{ 0 };
				if (random)
					sample_choice = SeleSample(); // Generate a random number
				else
					sample_choice = i * batch + bch; // Sequential retrieve

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

					std::string obj_name{ boost::str(boost::format("%|04|_00") % (smp+1)) };
					std::string obj_full_path{objs_path + obj_name + ".obj" };

					std::cout << obj_full_path << std::endl;

					bool is_success{ false }; // Flag indicating if forward simulation success
					bar_progress += bar_step;
					bar.set_option(option::PostfixText{ sample_step });
					bar.set_progress(bar_progress);

					if (smp == 0 && all_steps)
						clothStateCurrt = simnet.forward(clothStateCurrt, is_success); // Initial sample: set cloth state
					else
						clothStateCurrt = simnet.forward(is_success); // Later samples: directly moving on

					if (!is_success) break; // Break out from the loop if NaN detected in simulating

					++success_steps;

					Tensor GT_LPos{ torch::zeros_like(clothStateCurrt.LPos) }; // Ground Truth Lagrangian Position
					load_obj_train(obj_full_path, GT_LPos);

					Tensor Dis_L{ clothStateCurrt.LPos - GT_LPos }; // Distance Lagrangian Position

					Tensor L2_Dis_L{ torch::bmm(Dis_L.transpose(2,1), Dis_L) }; // L2 Distance of nodes' Lagrangian Position

					Tensor PosLoss{ torch::mean(L2_Dis_L) }; // Mean of L2 Position distance at one step

					loss = loss + PosLoss;
				}
			}
			loss = loss / ((success_steps - 1) > 0 ? (success_steps - 1) : 1);

			loss.backward(); // compute grad of register paremeters

			std::cout << "\nrho1 grad" << simnet.rho1.grad() << std::endl;
			std::cout << "rho2 grad" << simnet.rho2.grad() << std::endl;

			std::cout << "Y1 grad" << simnet.Y1.grad() << std::endl;
			std::cout << "Y2 grad" << simnet.Y2.grad() << std::endl;

			std::cout << "B1 grad" << simnet.B1.grad() << std::endl;
			std::cout << "B2 grad" << simnet.B2.grad() << std::endl;

			std::cout << "S grad" << simnet.S.grad() << std::endl;

			bool valid_step{ true };

			for (const auto& para : simnet.parameters())
				if (torch::isnan(para.grad()).item<bool>() ||
					torch::isinf(para.grad()).item<bool>()) {
					valid_step = false;
					break;
				}

			if (valid_step)
				cloth_opt.step(); // Excute Optim

			auto rho1_constraint = constraint_para(simnet.rho1, rho1_range.first, rho1_range.second);
			auto rho2_constraint = constraint_para(simnet.rho2, rho2_range.first, rho2_range.second);

			auto Y1_constraint = constraint_para(simnet.Y1, Y1_range.first, Y1_range.second);
			auto Y2_constraint = constraint_para(simnet.Y2, Y2_range.first, Y2_range.second);

			auto B1_constraint = constraint_para(simnet.B1, B1_range.first, B1_range.second);
			auto B2_constraint = constraint_para(simnet.B2, B2_range.first, B2_range.second);

			auto S_constraint = constraint_para(simnet.S, S_range.first, S_range.second);

			std::cout << "\nOptimized rho1: " << rho1_constraint.item<double>() << std::endl;
			std::cout << "Optimized rho2: " << rho2_constraint.item<double>() << std::endl;

			std::cout << "Optimized Y1: " << Y1_constraint.item<double>() << std::endl;
			std::cout << "Optimized Y2: " << Y2_constraint.item<double>() << std::endl;

			std::cout << "Optimized B1: " << B1_constraint.item<double>() << std::endl;
			std::cout << "Optimized B2: " << B2_constraint.item<double>() << std::endl;

			std::cout << "Optimized S: " << S_constraint.item<double>() << std::endl;

			std::cout << "Loss: " << loss.item<double>() << std::endl;

			std::map<std::string, Tensor> constrained_para;

			constrained_para.insert({ "rho1", rho1_constraint });
			constrained_para.insert({ "rho2", rho2_constraint });

			constrained_para.insert({ "Y1", Y1_constraint });
			constrained_para.insert({ "Y2", Y2_constraint });

			constrained_para.insert({ "B1", B1_constraint });
			constrained_para.insert({ "B2", B2_constraint });

			constrained_para.insert({ "S", S_constraint });

			SaveTrainProc(file, curr_step, train_steps, loss, constrained_para); // save loss and parameters
		}
	}
	file.close();
}