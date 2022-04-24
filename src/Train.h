#pragma once
#include "YarnSimNet.h"
#include "Saver.h"
#include "Random.h"
#include "indicators/progress_bar.hpp"
#include <cmath>


using namespace indicators;

void train(std::string save_path, std::string data_path, int epoch, int iter, int batch, int sample, int steps, bool random) {

	// epoch: the number of training epoches
	// iter: 
	// 1.if not random, the number of bacthes in each iteration 
	// (save in variable iteration) (see all training data)
	// 2. if random, use the give iter as iteration.
	// batch: the number of samples in each batch
	// sample: the number of steps in each sample
	// steps : the number of simulation steps in traing data

	// file in which saves training result
	H5File file{save_path, H5F_ACC_TRUNC};

	int samples{ steps / sample }; // the number of sample of training dataset
	int iteration{ random? iter: (samples / batch) }; // the number of training iteration
	int train_steps{ epoch * iteration }; // the number of training steps (backprop steps)

	RandInt SeleSample{ 0, samples - 1 }; // For Randomly Select Sample

	auto SizePhy = cloth_property_loader(data_path); // Read Cloth Size and Cloth Physical Properties

	H5ClothSize size{ SizePhy.first };
	H5Physical phy{ SizePhy.second };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(9, 0));
	
	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	// Initialization Parameters
	double init_rho = 0.001; // 0.002
	double init_B = 0.0002; // 0.0001
	double init_Y{ 9500.0 }; // 10000
	//double init_S{ 0.5 }; 
	//double init_df{ 500 };
	//double init_kf{ 500 };
	//double init_mu{ 0.7 };

	auto rho_range = std::make_pair(0.003, 0.0);
	auto Y_range = std::make_pair(12000.0, 8000.0);
	auto B_range = std::make_pair(0.0003, 0.0);

	phy.rho = invert_constraint_para(init_rho, rho_range.first, rho_range.second);
	phy.Y = invert_constraint_para(init_Y, Y_range.first, Y_range.second);
	phy.B = invert_constraint_para(init_B, B_range.first, B_range.second);

	std::cout << "Epoch: " << std::to_string(epoch) << std::endl;
	std::cout << "Iteration: " << std::to_string(iteration) << std::endl;
	std::cout << "Batch Size: " << std::to_string(batch) << std::endl;
	std::cout << "Sample Size: " << std::to_string(sample - 1) << std::endl;
	std::cout << "Random Sampling: " << (random ? "True" : "False") << std::endl;
	
	YarnSimNet simnet(size, phy, handles, h, rho_range, Y_range, B_range); // Construct Net

	torch::optim::OptimizerParamGroup rho({ simnet.named_parameters()["rho"] });
	torch::optim::OptimizerParamGroup Y({ simnet.named_parameters()["Y"] });
	torch::optim::OptimizerParamGroup B({ simnet.named_parameters()["B"] });
	torch::optim::Adam cloth_opt({rho, Y, B}, torch::optim::AdamOptions(0.1)); // Construct Optimizor
	static_cast<torch::optim::AdamOptions&>(cloth_opt.param_groups()[0].options()).lr(0.23);
	static_cast<torch::optim::AdamOptions&>(cloth_opt.param_groups()[1].options()).lr(0.12);
	static_cast<torch::optim::AdamOptions&>(cloth_opt.param_groups()[2].options()).lr(0.15);

	for (int epc = 0; epc < epoch; ++epc) {
		std::cout << "Epoch: " << epc << std::endl;
		for (int i = 0; i < iteration; ++i) {
			// Update parameters each batch
			std::cout << "Batch: " << i << std::endl;
			int success_steps{ 0 }; // Success simulation steps
			int curr_step{ epc * iteration + i  }; // cuttent step in all steps
			Tensor loss{ torch::zeros(1, opts) };	// Create loss 

			simnet.reset_cloth();
			cloth_opt.zero_grad();

			for (int bch = 0; bch < batch; ++bch) {
				std::cout << "Sample: " << std::to_string(bch+1) << std::endl;
				
				int rand_choice{ SeleSample()}; // Generate a random number

				ClothState clothStateSample = data_loader(data_path, rand_choice * sample, rand_choice * sample + sample, opts); // Load training batch
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

					bool is_success{false}; // Flag indicating if forward simulation success
					bar_progress += bar_step;
					bar.set_option(option::PostfixText{ sample_step });
					bar.set_progress(bar_progress);

					if (smp == 0)
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

					Tensor GT_LVel{ clothStateSample.LVel[smp + 1] }; // Ground Truth Lagrangian Velocity;
					Tensor GT_EVel{ clothStateSample.EVel[smp + 1] }; // Ground Truth Euler Velocity;

					Tensor Diff_LVel{ clothStateCurrt.LVel - GT_LVel }; // Lagrangian Velocity Difference 
					Tensor Diff_EVel{ clothStateCurrt.EVel - GT_EVel }; // Euler Velocity Difference

					Tensor L2_Diff_LVel{ torch::bmm(Diff_LVel.transpose(2,1), Diff_LVel) }; // L2 Distance of nodes' Lagrangian Velocity
					Tensor L2_Diff_EVel{ torch::bmm(Diff_EVel.transpose(2,1), Diff_EVel) }; // L2 Distance of nodes' Euler Velocity
					
					Tensor PosLoss{ torch::mean(L2_Dis_L) + torch::mean(L2_Dis_E) }; // Mean of L2 Position distance at one step

					Tensor VelLoss{ torch::mean(L2_Diff_LVel) + torch::mean(L2_Diff_EVel) }; // Mean of L2 Velocity difference at one step

					loss = loss + PosLoss + VelLoss; // Accumulate loss of a sample consists of multiple (=sample) steps
				}			
			}
			loss = loss / ((success_steps - 1) > 0 ? (success_steps - 1) : 1); // Mean of loss of one sample (sample-1 losses)
			loss.backward(); // compute grad of register paremeters

			std::cout << simnet.rho.grad() << std::endl;
			std::cout << simnet.Y.grad() << std::endl;
			std::cout << simnet.B.grad() << std::endl;
			//std::cout << simnet.S.grad() << std::endl;
			//std::cout << simnet.kf.grad() << std::endl;
			//std::cout << simnet.mu.grad() << std::endl;

			if (!torch::isnan(simnet.Y.grad()).item<bool>() && 
				!torch::isinf(simnet.Y.grad()).item<bool>() &&
				!torch::isnan(simnet.B.grad()).item<bool>() &&
				!torch::isinf(simnet.B.grad()).item<bool>() &&
				!torch::isnan(simnet.rho.grad()).item<bool>() &&
				!torch::isinf(simnet.rho.grad()).item<bool>()) {
				cloth_opt.step(); // Excute Optim
			}

			auto rho_constrained = constraint_para(simnet.rho, rho_range.first, rho_range.second);
			auto Y_constrained = constraint_para(simnet.Y, Y_range.first, Y_range.second);
			auto B_constrained = constraint_para(simnet.B, B_range.first, B_range.second);

			std::cout << "Loss: " << loss.item<double>() << std::endl;

			std::cout << "Constrained rho: " << rho_constrained.item<double>() << std::endl;
			std::cout << "Constrained B: " << Y_constrained.item<double>() << std::endl;
			std::cout << "Constrained Y: " << B_constrained.item<double>() << std::endl;
			//std::cout << "Constrainted S: " << constraint_S.item<double>() << std::endl;
			//std::cout << "Constrainted df: " << constraint_df.item<double>() << std::endl;
			//std::cout << "Constrainted kf: " << constraint_kf.item<double>() << std::endl;
			//std::cout << "Constrainted mu: " << constraint_mu.item<double>() << std::endl;

			std::map<std::string, Tensor> constrained_para;

			constrained_para.insert({ "rho", rho_constrained });
			constrained_para.insert({ "B", Y_constrained });
			constrained_para.insert({ "Y", B_constrained });
			//constrained_para.insert({ "S", constraint_S });
			//constrained_para.insert({ "df", constraint_df });
			//constrained_para.insert({ "kf", constraint_kf });
			//constrained_para.insert({ "mu", constraint_mu });
			
			SaveTrainProc(file, curr_step, train_steps, loss, constrained_para); // save loss and parameters
		}
	}
	file.close();	
}
