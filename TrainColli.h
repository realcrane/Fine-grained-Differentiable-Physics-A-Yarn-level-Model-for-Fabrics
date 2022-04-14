#pragma once
#pragma once
#include "YarnSimNetColli.h"
#include "YarnSimNet.h"
#include "Saver.h"
#include "Random.h"
#include "indicators/progress_bar.hpp"
#include "boost/format.hpp"
#include <cmath>

using namespace indicators;

void train_collision(std::string save_path, std::string data_path, 
	int epoch, int iter, int batch, int start_batch, int sample, 
	int steps, bool all_steps, bool random, bool para,
	std::string ground_path, std::string table_path) {

	// epoch: the number of training epoches
	// iter: 
	// 1.if not random, the number of bacthes in each iteration 
	// (save in variable iteration) (see all training data)
	// 2. if random, use the give iter as iteration.
	// batch: the number of samples in each batch
	// sample: the number of steps in each sample
	// steps : the number of simulation steps in training data

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

	auto SizePhy = cloth_property_loader(data_path); // Read Cloth Size and Cloth Physical Properties

	H5ClothSize size{ SizePhy.first };
	double height{ 0.025 };
	H5Physical phy{ SizePhy.second };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(4, 0));

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	// Initialization Parameters
	// Parameters for training
	double init_rho = 0.002; // 0.002
	double init_Y = 550000.0; // 500000
	double init_B = 0.00014; // 0.00014

	auto rho_range = std::make_pair(0.003, 0.001);
	auto Y_range = std::make_pair(600000.0, 300000.0);
	auto B_range = std::make_pair(0.0002, 0.00005);

	phy.rho = invert_constraint_para(init_rho, rho_range.first, rho_range.second);
	phy.Y = invert_constraint_para(init_Y, Y_range.first, Y_range.second);
	phy.B = invert_constraint_para(init_B, B_range.first, B_range.second);

	std::cout << "Epoch: " << std::to_string(epoch) << std::endl;
	std::cout << "Iteration: " << std::to_string(iteration) << std::endl;
	std::cout << "Batch Size: " << std::to_string(batch) << std::endl;
	std::cout << "Sample Size: " << std::to_string(sample - 1) << std::endl;
	std::cout << "Random Sampling: " << (random ? "True" : "False") << std::endl;
	std::cout << "Parallel: " << (para ? "True" : "False") << std::endl;

	YarnSimNetColli simnet(size, height, phy, handles, h, para, rho_range, Y_range, B_range, ground_path, table_path); // Construct Net Train

	//torch::optim::OptimizerParamGroup rho({ simnet.named_parameters()["rho"] });
	//torch::optim::OptimizerParamGroup Y({ simnet.named_parameters()["Y"] });
	//torch::optim::OptimizerParamGroup B({ simnet.named_parameters()["B"] });

	//torch::optim::SGD cloth_opt({ rho, Y, B }, torch::optim::SGDOptions(1e5));

	//static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[0].options()).lr(38e5);
	//static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[1].options()).lr(13e5);
	//static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[2].options()).lr(11e6);

	torch::optim::Adam cloth_opt(simnet.parameters(), torch::optim::AdamOptions(0.1));

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
					sample_choice = i * batch + bch + start_batch; // Sequential retrieve

				std::cout << "Sample Choice: " << sample_choice << std::endl;

				ClothState clothStateSample = data_loader(data_path, sample_choice * sample , sample_choice * sample + sample, opts); // Load training batch
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

					if (smp == 0 || smp == all_steps)
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

					std::cout << "Loss: " << loss << std::endl;

					loss = loss + PosLoss; // Accumulate loss of a sample consists of multiple (=sample) steps
				}
			}
			loss = loss / ((success_steps - 1) > 0 ? (success_steps - 1) : 1); // Mean of loss of one sample (sample-1 losses)	

			loss.backward(); // compute grad of register paremeters

			bool valid_step{ true };

			std::cout << "\nrho grad" << simnet.rho.grad() << std::endl;
			std::cout << "Y grad" << simnet.Y.grad() << std::endl;
			std::cout << "B grad" << simnet.B.grad() << std::endl;

			for (const auto& para : simnet.parameters())
				if (torch::isnan(para.grad()).item<bool>() ||
					torch::isinf(para.grad()).item<bool>()) {
					valid_step = false;
					break;
				}

			if (valid_step)
				cloth_opt.step(); // Excute Optim

			auto rho_constraint = constraint_para(simnet.rho, rho_range.first, rho_range.second);
			auto Y_constraint = constraint_para(simnet.Y, Y_range.first, Y_range.second);
			auto B_constraint = constraint_para(simnet.B, B_range.first, B_range.second);

			std::cout << "\nOptimized rho1: " << rho_constraint.item<double>() << std::endl;
			std::cout << "Optimized Y1: " << Y_constraint.item<double>() << std::endl;
			std::cout << "Optimized B1: " << B_constraint.item<double>() << std::endl;

			std::cout << "Loss: " << loss.item<double>() << std::endl;

			std::map<std::string, Tensor> constrained_para;

			constrained_para.insert({ "rho", rho_constraint });
			constrained_para.insert({ "Y", Y_constraint });
			constrained_para.insert({ "B", B_constraint });

			SaveTrainProc(file, curr_step, train_steps, loss, constrained_para); // save loss and parameters
		}
	}
	file.close();
}
