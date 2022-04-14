#include <iostream>
#include <limits>
#include "Train.h"
#include "TrainHomo.h"
#include "TrainControl.h"
#include "TrainHeter.h"
#include "TrainColli.h"
#include "TestHeter.h"
#include "TestHomo.h"
#include "Simulate.h"
#include "Solvers.h"
#include "TrainEmbed.h"

void TrainHomo() {

	std::string data_path = "./data/sim_homo.h5";
	std::string save_path = "./train/HangHeter.h5";

	train(save_path, data_path, 10, 20, 1, 10, 100, false);
}

void TrainControl()
{
	int epoch{ 70 };
	int frame_num{ 99 };
	int steps{ 100 };

	bool is_all_steps{ false };
	bool is_random{ false };
	bool is_para{ true };

	std::string data_path = "./data/Control_Plain_Drop_100Steps.h5";
	std::string save_path = "./train/Control_Test.h5";

	train_control(save_path, data_path, epoch, 1, 1, frame_num + 1, steps, is_all_steps, is_random);
}

void TrainEmbed()
{
	int epoch{ 70 };
	int frame_num{ 99 };
	int steps{ 100 };

	bool is_all_steps{ false };
	bool is_random{ false };
	bool is_para{ true };

	std::string data_path = "./data/Control_Plain_Drop_100Steps.h5";
	std::string save_path = "./train/Embed_Control_New.h5";

	train_embed(save_path, data_path, epoch, 1, 1, frame_num + 1, steps, is_all_steps, is_random);
}

void TrainHeterFew() {
	// The parameters controlling training
	int epochs{ 70 };  // Training Epochs
	int frame_num{ 25 }; // Training frames
	int steps{ 500 }; // Simulation steps in the give training data
	bool wordy{ true }; // if true, print parameters' grad and values every epoch

	// Path of training data
	std::string data_path{ "./NewTrainingData/size5/Wind_Cotton_Poly_Plain_Size5.h5" };	
	// Set file saving training result
	std::string save_path{ "./train/test.h5" };
	// Go into the following function to set parameters' learning rate
	train_heter_few(save_path, data_path, epochs, 1, 1, frame_num + 1, steps, false, false, wordy);
}

void TrainHeterFull() {
	// The parameters controlling training
	int epochs{ 70 };  // Training Epochs
	int frame_num{ 5 }; // Training frames
	int steps{ 500 }; // Simulation steps in the give training data
	bool wordy{ true }; // if true, print parameters' grad and values every epoch

	// Path of training data
	std::string data_path{ "./NewTrainingData/NewShear/size17/Wind_Cotton_Poly_Plain_Size17.h5" };
	// Set file saving training result
	std::string save_path{ "./train/cotton_poly_plain_wind_no_constraint.h5" };
	// Go into the following function to set parameters' learning rate
	train_heter_full(save_path, data_path, epochs, 1, 1, frame_num + 1, steps, false, false, wordy);
}

void TrainCollision() {

	std::string data_path{ "./data/collision.h5" };
	std::string save_path{ "./train/Test.h5" };

	std::string ground_path{ "./obs/ground.obj" };
	std::string table_path{ "./obs/wall.obj" };

	train_collision(save_path, data_path, 1, 1, 1, 0, 16, 20, false, false, true, ground_path, table_path);
}

void TrainFromSheet() {
	// The parameters controlling training
	int epochs{ 70 };  // Training Epochs
	int frame_num{ 6 }; // Training frames
	int steps{ 100 }; // Simulation steps in the give training data

	// Load training data where the model can find cloth properties: size, initial pose, woven pattern, etc.
	std::string data_path{ "./data/Hang_Satin_100Steps.h5" };
	std::string save_path{ "./train/Test.h5" };  // Path of saved file that record training procedure
	std::string render_path{ "./train_render/" }; // No need to be specified, it's for debugging
	std::string objs_path{ "./NewTrainingData/Sheet/white_dot_on_blk/out/" };  // Training data path.

	train_homo_obj(save_path, data_path, render_path, objs_path, epochs, 1, 1, frame_num, steps, false, false, true);
}

void TestingHeter() {
	std::string data_path{ "./NewTrainingData/size17/cotton_poly_satin/Wind_Cotton_Poly_Satin_Size17.h5" };
	std::string objs_path{ "./NewTrainingData/Sheet/cotton_poly_satin_25steps/" };
	std::string save_path{ "./Experiments/TrainFromSheetTesting/cotton_poly_satin_25steps.txt" };

	test_heter(save_path, data_path, objs_path, 51, 500);
}

void TestingHeterH5() {
	int testing_steps{ 51 };
	// You need to set the path of the ground truth data
	std::string data_path{ "./NewTrainingData/size17/poly_silk_plain/Wind_Poly_Silk_Plain_Size17.h5" };
	// You need to specify the location where the testing result should be saved.
	std::string save_path{ "./Experiments/TrainFromSheetTesting/test.txt" };

	test_heter_h5(save_path, data_path, testing_steps, 500);
}

void TestingHomo() {
	int testing_steps{ 51 };

	std::string data_path{ "./data/Hang_Satin_100Steps.h5" };
	// Ground Truch .obj Path
	std::string objs_path{ "./NewTrainingData/Sheet/white_dot_on_blk/out/" };
	// Path For Saving Testing Results
	std::string save_path{ "./Experiments/TrainFromSheetTesting/white_dots_on_blk_plain_25steps.txt" };

	test_homo(save_path, data_path, objs_path, testing_steps, 500);
}

int main() {

	// +Simulation related functions
	 
	//SimHomo();  // Simulate Pure Woven

	//SimHeter();  // Simulate Blend Woven

	//SimCollision();	// Simulate Collision

	//SimControl();


	// +Training 

	 //TrainHomo(); 

	 //TrainHeterFew();

	 //TrainHeterFull();

	 //TrainControl();

	 //TrainEmbed();

	 //TrainFromSheet();

	 //TrainCollision();


	// +Testing

	 //TestingHeterH5(); 

	 //TestingHomo();

	 //TestingHeter(); 

}