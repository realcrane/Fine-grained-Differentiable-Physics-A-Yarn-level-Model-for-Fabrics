#pragma once
#include "YarnSimNet.h"
#include "Saver.h"
#include "Random.h"
#include "indicators/progress_bar.hpp"
#include <cmath>

using namespace indicators;


void train_control(std::string save_path, std::string data_path, int epoch, int iter, int batch, int sample, int steps,
	bool all_steps, bool random) {

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

	std::vector<std::pair<int, int>> handles{};  // Handle vector

	handles.push_back(std::make_pair(0, 0));
	handles.push_back(std::make_pair(4, 0));
	handles.push_back(std::make_pair(0, 4));
	handles.push_back(std::make_pair(4, 4));

	Tensor h = torch::tensor({ 0.001 }, opts); // Simulation Time Step Size: sec

	// Initialization Parameters
	double lift_force_strength_one = 0.00161396;
	double drag_force_strength_one = 0.00093096;

	double lift_force_strength_two = 0.00166182;
	double drag_force_strength_two = 0.00096396;

	double lift_force_strength_three = 0.00104908;
	double drag_force_strength_three = 0.00043926;

	double lift_force_strength_four = 0.00158723;
	double drag_force_strength_four = 0.00178333;

	double lift_force_strength_five = 0.00081228;
	double drag_force_strength_five = 0.00132253;

	double lift_force_strength_six = 0.00053757;
	double drag_force_strength_six = 0.00172603;

	int force_stop_frame = 5;

	std::cout << "Epoch: " << std::to_string(epoch) << std::endl;
	std::cout << "Iteration: " << std::to_string(iteration) << std::endl;
	std::cout << "Batch Size: " << std::to_string(batch) << std::endl;
	std::cout << "Sample Size: " << std::to_string(sample - 1) << std::endl;
	std::cout << "Random Sampling: " << (random ? "True" : "False") << std::endl;

	YarnSimNetControl simnet(size, phy, 
		lift_force_strength_one, drag_force_strength_one, 
		lift_force_strength_two, drag_force_strength_two, 
		lift_force_strength_three, drag_force_strength_three,
		lift_force_strength_four, drag_force_strength_four,
		lift_force_strength_five, drag_force_strength_five,
		lift_force_strength_six, drag_force_strength_six,
		handles, h); // Construct Net

	torch::optim::OptimizerParamGroup lift_force_one({ simnet.named_parameters()["lift_force_one"] });
	torch::optim::OptimizerParamGroup drag_force_one({ simnet.named_parameters()["drag_force_one"] });
	torch::optim::OptimizerParamGroup lift_force_two({ simnet.named_parameters()["lift_force_two"] });
	torch::optim::OptimizerParamGroup drag_force_two({ simnet.named_parameters()["drag_force_two"] });
	torch::optim::OptimizerParamGroup lift_force_three({ simnet.named_parameters()["lift_force_three"] });
	torch::optim::OptimizerParamGroup drag_force_three({ simnet.named_parameters()["drag_force_three"] });
	torch::optim::OptimizerParamGroup lift_force_four({ simnet.named_parameters()["lift_force_four"] });
	torch::optim::OptimizerParamGroup drag_force_four({ simnet.named_parameters()["drag_force_four"] });
	torch::optim::OptimizerParamGroup lift_force_five({ simnet.named_parameters()["lift_force_five"] });
	torch::optim::OptimizerParamGroup drag_force_five({ simnet.named_parameters()["drag_force_five"] });
	torch::optim::OptimizerParamGroup lift_force_six({ simnet.named_parameters()["lift_force_six"] });
	torch::optim::OptimizerParamGroup drag_force_six({ simnet.named_parameters()["drag_force_six"] });

	torch::optim::SGD cloth_opt({ 
		lift_force_one, drag_force_one, 
		lift_force_two, drag_force_two,
		lift_force_three, drag_force_three,
		lift_force_four, drag_force_four,
		lift_force_five, drag_force_five,
		lift_force_six, drag_force_six },
		torch::optim::SGDOptions(0.0001));

	ClothState clothStateInit = data_loader(data_path, 0, 1, opts); // Load cloth initial state

	ClothState clothStateEnd = data_loader(data_path, steps - 1, steps, opts); // Load cloth goal state

	ClothState clothStateGoal{
		clothStateEnd.LPos[0],
		clothStateEnd.LPosFix[0],
		clothStateEnd.LVel[0],
		clothStateEnd.EPos[0],
		clothStateEnd.EPosBar[0],
		clothStateEnd.EVel[0] };

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

			ClothState clothStateCurrt{
				clothStateInit.LPos[0],
				clothStateInit.LPosFix[0],
				clothStateInit.LVel[0],
				clothStateInit.EPos[0],
				clothStateInit.EPosBar[0],
				clothStateInit.EVel[0] }; // reset cloth initial state

			for (int bch = 0; bch < batch; ++bch) {
				std::cout << "Sample: " << std::to_string(bch + 1) << std::endl;

				int sample_choice{ 0 };
				if (random)
					sample_choice = SeleSample(); // Generate a random number
				else
					sample_choice = i * batch + bch; // Sequential retrieve
				
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

					std::string objname = boost::str(boost::format("Epoch%|02|_Cloth%|04|") % epc % smp);
					std::string save_path = "./train_objs/" + objname + ".obj"; // Path of .obj files

					bool is_success{ false }; // Flag indicating if forward simulation success
					bar_progress += bar_step;
					bar.set_option(option::PostfixText{ sample_step });
					bar.set_progress(bar_progress);

					if ( smp == 0 )
						clothStateCurrt = simnet.forward(clothStateCurrt, is_success, smp); 
					else if ( smp <= force_stop_frame )
						clothStateCurrt = simnet.forward(is_success, save_path, smp); 
					else
						clothStateCurrt = simnet.forward(is_success, save_path, 6);

					if (!is_success)
					{
						std::cout << "Crash Simulation" << std::endl;
						break; // Break out from the loop if NaN detected in simulating
					}

					++success_steps;

				}
			}

			Tensor GT_LPos{ clothStateGoal.LPos }; // Ground Truth Lagrangian Position

			Tensor mass_center_distance{ (torch::mean(GT_LPos, 0) - torch::mean(clothStateCurrt.LPos, 0)).squeeze() };

			Tensor mean_distance_L{ torch::matmul(mass_center_distance, mass_center_distance) };

			loss = mean_distance_L;

			std::cout << "Loss: " << loss.item<double>() << std::endl;

			loss.backward(); // compute grad of register paremeters

			std::cout << simnet.lift_force_one.grad() << std::endl;
			std::cout << simnet.drag_force_one.grad() << std::endl;
			std::cout << simnet.lift_force_two.grad() << std::endl;
			std::cout << simnet.drag_force_two.grad() << std::endl;
			std::cout << simnet.lift_force_three.grad() << std::endl;
			std::cout << simnet.drag_force_three.grad() << std::endl;
			std::cout << simnet.lift_force_four.grad() << std::endl;
			std::cout << simnet.drag_force_four.grad() << std::endl;
			std::cout << simnet.lift_force_five.grad() << std::endl;
			std::cout << simnet.drag_force_five.grad() << std::endl;
			std::cout << simnet.lift_force_six.grad() << std::endl;
			std::cout << simnet.drag_force_six.grad() << std::endl;

			for (auto p : simnet.parameters())
			{
				if (!torch::isnan(p).item<bool>() && !torch::isinf(p).item<bool>()) 
					cloth_opt.step(); // Excute Optim
			}

			std::cout << "Optimized Lift Force One: " << simnet.lift_force_one.item<double>() << std::endl;
			std::cout << "Optimized Drag Force One: " << simnet.drag_force_one.item<double>() << std::endl;

			std::cout << "Optimized Lift Force Two: " << simnet.lift_force_two.item<double>() << std::endl;
			std::cout << "Optimized Drag Force Two: " << simnet.drag_force_two.item<double>() << std::endl;

			std::cout << "Optimized Lift Force Three: " << simnet.lift_force_three.item<double>() << std::endl;
			std::cout << "Optimized Drag Force Three: " << simnet.drag_force_three.item<double>() << std::endl;

			std::cout << "Optimized Lift Force Four: " << simnet.lift_force_four.item<double>() << std::endl;
			std::cout << "Optimized Drag Force Four: " << simnet.drag_force_four.item<double>() << std::endl;

			std::cout << "Optimized Lift Force Five: " << simnet.lift_force_five.item<double>() << std::endl;
			std::cout << "Optimized Drag Force Five: " << simnet.drag_force_five.item<double>() << std::endl;

			std::cout << "Optimized Lift Force Six: " << simnet.lift_force_six.item<double>() << std::endl;
			std::cout << "Optimized Drag Force Six: " << simnet.drag_force_six.item<double>() << std::endl;

			std::map<std::string, Tensor> constrained_para;

			constrained_para.insert({ "lift_one", simnet.lift_force_one });
			constrained_para.insert({ "drag_one", simnet.drag_force_one });

			constrained_para.insert({ "lift_two", simnet.lift_force_two });
			constrained_para.insert({ "drag_two", simnet.drag_force_two });

			constrained_para.insert({ "lift_three", simnet.lift_force_three });
			constrained_para.insert({ "drag_three", simnet.drag_force_three });

			constrained_para.insert({ "lift_four", simnet.lift_force_four });
			constrained_para.insert({ "drag_four", simnet.drag_force_four });

			constrained_para.insert({ "lift_five", simnet.lift_force_five });
			constrained_para.insert({ "drag_five", simnet.drag_force_five });

			constrained_para.insert({ "lift_six", simnet.lift_force_six });
			constrained_para.insert({ "drag_six", simnet.drag_force_six });

			SaveTrainProc(file, curr_step, train_steps, loss, constrained_para); // save loss and parameters
		}
	}
	file.close();
}

