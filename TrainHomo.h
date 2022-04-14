#pragma once
#include "YarnSimNetHomo.h"
#include "Saver.h"
#include "Random.h"
#include "indicators/progress_bar.hpp"
#include "boost/format.hpp"
#include <cmath>

using namespace indicators;

void train_homo_obj(std::string save_path, std::string data_path, 
	std::string render_path, std::string objs_path,
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
	//torch::NoGradGuard no_grad;

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
	H5Physical phy{ SizePhy.second };

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(16, 0));

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	// Initialization Parameters

	double init_rho{ 0.004 };
	double init_Y{ 1e6 };
	double init_B{ 0.0001 };
	double init_S{ 20000.0 };

	auto rho_range = std::make_pair(0.008, 0.001);
	auto Y_range = std::make_pair(2000000.0, 0.0);
	auto B_range = std::make_pair(0.0002, 0.0);
	auto S_range = std::make_pair(30000, 0);

	phy.rho = invert_constraint_para(init_rho, rho_range.first, rho_range.second);
	phy.Y = invert_constraint_para(init_Y, Y_range.first, Y_range.second);
	phy.B = invert_constraint_para(init_B, B_range.first, B_range.second);
	phy.S = invert_constraint_para(init_S, S_range.first, S_range.second);

	std::cout << "Epoch: " << std::to_string(epoch) << std::endl;
	std::cout << "Iteration: " << std::to_string(iteration) << std::endl;
	std::cout << "Batch Size: " << std::to_string(batch) << std::endl;
	std::cout << "Sample Size: " << std::to_string(sample - 1) << std::endl;
	std::cout << "Random Sampling: " << (random ? "True" : "False") << std::endl;
	std::cout << "Parallel: " << (para ? "True" : "False") << std::endl;

	YarnSimNetHomo simnet(size, phy, handles, h, rho_range, Y_range, B_range, S_range); // Construct Net Train

	torch::optim::OptimizerParamGroup rho({ simnet.named_parameters()["rho"] });
	torch::optim::OptimizerParamGroup Y({ simnet.named_parameters()["Y"] });
	torch::optim::OptimizerParamGroup B({ simnet.named_parameters()["B"] });
	torch::optim::OptimizerParamGroup S({ simnet.named_parameters()["S"] });

	torch::optim::SGD cloth_opt({ rho, Y, B, S }, torch::optim::SGDOptions(1e5));

	// ***Alter learning rate***
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[0].options()).lr(19e6); // lr rho
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[1].options()).lr(8e8);  // lr Y
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[2].options()).lr(23e9); // lr B
	static_cast<torch::optim::SGDOptions&>(cloth_opt.param_groups()[3].options()).lr(78e7); // lr S

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

					std::string obj_name{ boost::str(boost::format("%|04|_00") % (smp + 1)) };
					std::string obj_full_path{ objs_path + obj_name + ".obj" };

					//std::string render_name{ boost::str(boost::format("Cloth%|04|") % smp) };
					//std::string render_full_path{ render_path + render_name + ".obj" };

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
					load_obj_train(obj_full_path, GT_LPos);

					Tensor Dis_L{ clothStateCurrt.LPos - GT_LPos }; // Distance Lagrangian Position

					Tensor L2_Dis_L{ torch::bmm(Dis_L.transpose(2,1), Dis_L) }; // L2 Distance of nodes' Lagrangian Position

					Tensor PosLoss{ torch::mean(L2_Dis_L) }; // Mean of L2 Position distance at one step

					loss = loss + PosLoss;
				}
			}
			loss = loss / ((success_steps - 1) > 0 ? (success_steps - 1) : 1);

			std::cout << "Loss: " << loss.item<double>() << std::endl;

			loss.backward(); // compute grad of register paremeters

			std::cout << "\nrho1 grad" << simnet.rho.grad() << std::endl;
			std::cout << "Y1 grad" << simnet.Y.grad() << std::endl;
			std::cout << "B1 grad" << simnet.B.grad() << std::endl;
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

			auto rho_constraint = constraint_para(simnet.rho, rho_range.first, rho_range.second);
			auto Y_constraint = constraint_para(simnet.Y, Y_range.first, Y_range.second);
			auto B_constraint = constraint_para(simnet.B, B_range.first, B_range.second);
			auto S_constraint = constraint_para(simnet.S, S_range.first, S_range.second);

			std::cout << "\nOptimized rho1: " << rho_constraint.item<double>() << std::endl;
			std::cout << "Optimized Y1: " << Y_constraint.item<double>() << std::endl;
			std::cout << "Optimized B1: " << B_constraint.item<double>() << std::endl;
			std::cout << "Optimized S: " << S_constraint.item<double>() << std::endl;

			std::cout << "Loss: " << loss.item<double>() << std::endl;

			std::map<std::string, Tensor> constrained_para;

			constrained_para.insert({ "rho", rho_constraint });
			constrained_para.insert({ "Y", Y_constraint });
			constrained_para.insert({ "B", B_constraint });
			constrained_para.insert({ "S", S_constraint });

			SaveTrainProc(file, curr_step, train_steps, loss, constrained_para); // save loss and parameters
		}
	}
	file.close();
}
