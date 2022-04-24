#pragma once
#include "Debugger.h"
#include "Loader.h"
#include "Collision.h"
#include "boost/format.hpp"
#include "indicators/block_progress_bar.hpp"

using namespace indicators;

void SimHomo() {
	// Parallel Simulation Pure Woven Cloth

	// Physical Parameters
	int width{ 10 }, length{ 10 };
	double height{ 5.0 };

	double L{ 0.001 }, R{ 0.0005 };

	Tensor G = torch::tensor({ 0.0, 0.0, 9.8 }, opts).view({ 3,1 });
	Tensor rho = torch::tensor({ 0.002 }, opts);
	Tensor Y = torch::tensor({ 500000 }, opts);
	Tensor B = torch::tensor({ 0.00014 }, opts);
	Tensor kc = torch::tensor({ 1.0 }, opts);
	Tensor mu = torch::tensor({ 0.5 }, opts);
	Tensor df = torch::tensor({ 1000.0 }, opts);
	Tensor kf = torch::tensor({ 1000.0 }, opts);
	Tensor S = torch::tensor({ 1000.0 }, opts); 
	Tensor handle_stiffness = torch::tensor({ 1e4 }, opts);

	Tensor wind_vel{ torch::tensor({0.0, 5.0, 0.0}, opts).view({3,1}) };
	Tensor wind_density{ torch::tensor({2.0}, opts) };
	Tensor wind_drag{ torch::tensor({0.5}, opts) };
	// Physical Parameters End

	Wind wind{ wind_vel , wind_density , wind_drag };

	Yarn yarn{R, rho, Y, B};

	std::vector<Yarn> yarns{ yarn };

	Cloth cloth(width, length, height, L, R, yarns, kc, kf, df, mu, S, handle_stiffness, WovenPattern::Plain, InitPose::Upright);

	Environment env{ &cloth, G, wind };

	PhysicsCloth phy{ &env };
	phy.InitMV();

	Tensor h = torch::tensor({ 0.001 }, opts);

	std::vector<std::pair<int, int>> handles;

	// *** Define handle nodes ***
	handles.push_back(std::make_pair(0, 0));
	// The following line should be set to
	// Simulation 5 x 5 cloth : handles.push_back(std::make_pair(4, 0));
	// Simulation 10 x 10 cloth : handles.push_back(std::make_pair(9, 0));
	// Simulation 17 x 17 cloth : handles.push_back(std::make_pair(16, 0));
	// Simulation 25 x 25 cloth : handles.push_back(std::make_pair(24, 0));
	handles.push_back(std::make_pair(9, 0));

	cloth.set_handles(handles);

	std::string h5path{ "./data/sim_homo_env.h5" }; // File Path of h5 file
	std::shared_ptr<H5File> file = std::make_shared<H5File>(h5path, H5F_ACC_TRUNC);

	int steps{ 100 };

	BlockProgressBar bar{
		option::BarWidth{50},
		option::Start{"["},
		option::End{"]"},
		option::ForegroundColor{Color::white},
		indicators::option::PostfixText{"Simulation Step: 0"},
		option::FontStyles{std::vector<FontStyle>{FontStyle::bold}} };
	float bar_step{ 100.0f / steps };
	float bar_progress{ 0.0f };

	for (int i = 0; i < steps; ++i) {
		std::string ind{ "Simulation Step: " + std::to_string(i + 1) };
		std::string objname = boost::str(boost::format("Cloth%|04|") % i);
		std::string save_path = "./objs/" + objname + ".obj"; // Path of .obj files
		//std::string save_path = "./objs_new/" + objname + ".obj"; // Path of .obj files

		SaveClothHomoYarn(cloth, env, file, steps, i);

		phy.FillMV();

		const Tensor A{ phy.get_SpaM() - (phy.get_SpaFPos()) * torch::pow(h, 2) - phy.get_SpaFVel() * h };

		const Tensor b{ h * (phy.get_SpaF() - torch::mm(phy.get_SpaFVel().transpose(0,1), phy.get_SpaVel())) + torch::mm(phy.get_SpaM(), phy.get_SpaVel()) };

		Tensor new_vel = EigenSolver::apply(A, b);

		cloth.update(new_vel, phy.F_E_is_Slides[0], h);

		phy.ResetMV();

		bar_progress += bar_step;
		bar.set_option(option::PostfixText{ ind });
		bar.set_progress(bar_progress);

		RenderCloth(cloth, save_path, RenderType::MeshTriangle);
	}

	file->close();
}

void SimCollision() {

	Obstacle table, ground;
	table.obsMesh.isCloth = false;
	ground.obsMesh.isCloth = false;

	table.density = torch::tensor({ 1e3 }, opts);
	ground.density = torch::tensor({ 1e3 }, opts);

	std::string table_path{ "./obs/wall.obj" };
	std::string ground_path{ "./obs/ground.obj" };

	load_obj(table_path, table.obsMesh);
	load_obj(ground_path, ground.obsMesh);

	table.obsMesh.add_adjecent();
	table.compute_mesh_mass();

	ground.obsMesh.add_adjecent();
	ground.compute_mesh_mass();

	double L{ 0.001 };
	double R{ 0.0005 };

	Tensor thickness{ torch::tensor({0.0005}, opts) };

	Tensor G = torch::tensor({ 0.0, 0.0, 9.8 }, opts).view({ 3,1 });
	Tensor rho = torch::tensor({ 0.002 }, opts);
	Tensor Y = torch::tensor({ 500000 }, opts);
	Tensor B = torch::tensor({ 0.00014 }, opts);
	Tensor kc = torch::tensor({ 1.0 }, opts);
	Tensor mu = torch::tensor({ 0.5 }, opts);
	Tensor kf = torch::tensor({ 1000.0 }, opts);
	Tensor df = torch::tensor({ 1000.0 }, opts);
	Tensor S = torch::tensor({ 1000.0 }, opts);
	Tensor handle_stiffness = torch::tensor({ 1e4 }, opts);

	Tensor wind_vel{ torch::tensor({0.0, 5.0, 0.0}, opts).view({3,1}) };
	Tensor wind_density{ torch::tensor({2.0}, opts) };
	Tensor wind_drag{ torch::tensor({0.5}, opts) };

	Wind wind{ wind_vel, wind_density, wind_drag };

	Yarn yarn{ R, rho, Y, B };

	std::vector<Yarn> yarns{ yarn };

	Cloth cloth(5, 5, 0.025, L, R, yarns, kc, kf, df, mu, S, handle_stiffness, WovenPattern::Plain, InitPose::Upright);

	Tensor h = torch::tensor({ 0.001 }, opts);

	Environment env{ &cloth, &table, &ground, G, wind };
	PhysicsCloth phy{ &env };
	phy.InitMV();

	std::string h5path{ "./data/collision.h5" };
	std::shared_ptr<H5File> file = std::make_shared<H5File>(h5path, H5F_ACC_TRUNC);

	std::vector<std::pair<int, int>> handles;

	// *** Define handle nodes ***
	handles.push_back(std::make_pair(0, 0));
	// The following line should be set to
	// Simulation 5 x 5 cloth : handles.push_back(std::make_pair(4, 0));
	// Simulation 10 x 10 cloth : handles.push_back(std::make_pair(9, 0));
	// Simulation 17 x 17 cloth : handles.push_back(std::make_pair(16, 0));
	// Simulation 25 x 25 cloth : handles.push_back(std::make_pair(24, 0));
	handles.push_back(std::make_pair(4, 0));

	cloth.set_handles(handles);

	Collision collision{ &cloth.clothMesh, &table.obsMesh, &ground.obsMesh, thickness, h, true };

	int steps{ 20 };

	for (int i = 0; i < steps; ++i) {
		std::cout << "Step: " << i << std::endl;
		std::string objname = boost::str(boost::format("Cloth%|04|") % i);
		std::string save_path = "./objs/" + objname + ".obj";
		SaveClothHomoYarn(cloth, env, file, steps, i);

		phy.FillMV();

		const Tensor A{ phy.get_SpaM() - (phy.get_SpaFPos()) * torch::pow(h, 2) - phy.get_SpaFVel() * h };
		const Tensor b{ h * (phy.get_SpaF() - torch::mm(phy.get_SpaFVel().transpose(0,1), phy.get_SpaVel())) + torch::mm(phy.get_SpaM(), phy.get_SpaVel()) };

		Tensor new_vel = EigenSolver::apply(A, b);

		cloth.update(new_vel, phy.F_E_is_Slides[0], h);
		phy.ResetMV();

		cloth.clothMesh.compute_ms_data();
		cloth.clothMesh.compute_ws_data();

		bool changed{ false };

		changed = collision.collision_response(0.1);

		if (changed) {
			cloth.update();
			cloth.clothMesh.compute_ms_data();
			cloth.clothMesh.compute_ws_data();
		}

		RenderCloth(cloth, save_path, RenderType::MeshTriangle);
	}

	file->close();
}

void SimControl() {
	// Simulation Pure Woven Cloth For Controlling

	// Physical Parameters
	int width{ 5 }, length{ 5 };
	double height{ 5.0 };

	double L{ 0.001 }, R{ 0.0005 };

	Tensor G = torch::tensor({ 0.0, 0.0, 9.8 }, opts).view({ 3,1 });
	Tensor rho = torch::tensor({ 0.002 }, opts);
	Tensor Y = torch::tensor({ 50000.0 }, opts);
	Tensor B = torch::tensor({ 0.00001 }, opts);
	Tensor kc = torch::tensor({ 1.0 }, opts);
	Tensor mu = torch::tensor({ 0.5 }, opts);
	Tensor df = torch::tensor({ 1000.0 }, opts);
	Tensor kf = torch::tensor({ 1000.0 }, opts);
	Tensor S = torch::tensor({ 1000.0 }, opts);
	Tensor handle_stiffness = torch::tensor({ 1e4 }, opts);

	Tensor wind_vel{ torch::tensor({0.0, 0.0, 0.0}, opts).view({3,1}) };
	Tensor wind_density{ torch::tensor({2.0}, opts) };
	Tensor wind_drag{ torch::tensor({0.5}, opts) };
	// Physical Parameters End

	Wind wind{ wind_vel , wind_density , wind_drag };

	Yarn yarn{ R, rho, Y, B };

	std::vector<Yarn> yarns{ yarn };

	Cloth cloth(width, length, height, L, R, yarns, kc, kf, df, mu, S, handle_stiffness, WovenPattern::Plain, InitPose::Flat);

	Environment env{ &cloth, G, wind };

	PhysicsCloth phy{ &env };
	phy.InitMV();

	Tensor h = torch::tensor({ 0.001 }, opts);

	std::vector<std::pair<int, int>> handles;

	cloth.set_handles(handles);

	std::string h5path{ "./data/Control_Plain_Drop_100Steps.h5" }; // File Path of h5 file
	std::shared_ptr<H5File> file = std::make_shared<H5File>(h5path, H5F_ACC_TRUNC);

	Tensor ext_force = torch::zeros({ 93, 1 }, opts);

	float scale = 1.0f;
	float ext_force_up_strength = 0.002f;
	float ext_force_forward_strength = 0.002f;

	// Drag Forward
	ext_force.index_put_({ 1, 0 }, ext_force_forward_strength * scale);
	ext_force.index_put_({ 13, 0 }, ext_force_forward_strength * scale);
	ext_force.index_put_({ 61, 0 }, ext_force_forward_strength * scale);
	ext_force.index_put_({ 73, 0 }, ext_force_forward_strength * scale);
	// Balance
	//ext_force.index_put_({ 0, 0 }, -ext_force_up_strength * scale);
	//ext_force.index_put_({ 6, 0 }, ext_force_up_strength * scale);
	//ext_force.index_put_({ 18, 0 }, -ext_force_up_strength * scale);
	//ext_force.index_put_({ 24, 0 }, ext_force_up_strength * scale);
	// Lifting
	ext_force.index_put_({ 2, 0 }, ext_force_up_strength * scale);
	ext_force.index_put_({ 14, 0 }, ext_force_up_strength * scale);
	ext_force.index_put_({ 62, 0 }, ext_force_up_strength * scale);
	ext_force.index_put_({ 74, 0 }, ext_force_up_strength * scale);

	int steps{ 100 };

	BlockProgressBar bar{
		option::BarWidth{50},
		option::Start{"["},
		option::End{"]"},
		option::ForegroundColor{Color::white},
		indicators::option::PostfixText{"Simulation Step: 0"},
		option::FontStyles{std::vector<FontStyle>{FontStyle::bold}} };
	float bar_step{ 100.0f / steps };
	float bar_progress{ 0.0f };

	for (int i = 0; i < steps; ++i) {
		std::string ind{ "Simulation Step: " + std::to_string(i + 1) };
		std::string objname = boost::str(boost::format("Cloth%|04|") % i);
		std::string save_path = "./objs/" + objname + ".obj"; // Path of .obj files

		SaveClothHomoYarn(cloth, env, file, steps, i);

		phy.FillMV();

		if (i > 5)
			ext_force = torch::zeros({ 93, 1 }, opts);

		const Tensor A{ phy.get_SpaM() - (phy.get_SpaFPos()) * torch::pow(h, 2) - phy.get_SpaFVel() * h };

		const Tensor b{ h * (phy.get_SpaF() + ext_force - torch::mm(phy.get_SpaFVel().transpose(0,1), phy.get_SpaVel())) + torch::mm(phy.get_SpaM(), phy.get_SpaVel()) };

		Tensor new_vel = EigenSolver::apply(A, b);

		cloth.update(new_vel, phy.F_E_is_Slides[0], h);

		phy.ResetMV();

		bar_progress += bar_step;
		bar.set_option(option::PostfixText{ ind });
		bar.set_progress(bar_progress);

		RenderCloth(cloth, save_path, RenderType::MeshTriangle);
	}

	file->close();
}


void SimHeter() {
	// Parallel Simulation Blend Woven
	// *** Set Cloth Size ***
	int width{ 10 }, length{ 10 };
	double height{ 5.0 };
	double L{0.001}, R{ 0.0005 };

	Tensor Init_LVel = torch::tensor( { 0.0, 0.0, 0.0 }, opts ).view({ 3,1 });

	Tensor G = torch::tensor({ 0.0, 0.0, 9.8 }, opts).view({ 3,1 });

	Tensor wind_vel{ torch::tensor({0.0, 5.0, 0.0}, opts).view({3,1}) };
	Tensor wind_density{ torch::tensor({2.0}, opts) };
	Tensor wind_drag{ torch::tensor({0.5}, opts) };

	Wind wind{ wind_vel , wind_density , wind_drag };

	Tensor handle_stiffness = torch::tensor({ 1e4 }, opts);

	// Yarn 1
	Tensor rho_cotton = torch::tensor({ 0.002 }, opts);
	Tensor Y_cotton = torch::tensor({ 500000 }, opts);
	Tensor B_cotton = torch::tensor({ 0.00014 }, opts);

	Yarn cotton_yarn{R, rho_cotton, Y_cotton, B_cotton};

	// Yarn 2
	Tensor rho_poly = torch::tensor({ 0.0025 }, opts);
	Tensor Y_poly = torch::tensor({ 170000 }, opts);
	Tensor B_poly = torch::tensor({ 0.00011 }, opts);

	Yarn poly_yarn{ R, rho_poly, Y_poly, B_poly };

	// Yarn 3
	Tensor rho_silk = torch::tensor({ 0.0024 }, opts);
	Tensor Y_silk = torch::tensor({ 120000 }, opts);
	Tensor B_silk = torch::tensor({ 0.00009 }, opts);

	Yarn silk_yarn{ R, rho_silk, Y_silk, B_silk };

	Tensor S1 = torch::tensor({ 1000.0 }, opts);  // Cotton Polyester
	Tensor S2 = torch::tensor({ 700 }, opts); // Cotton Silk
	Tensor S3 = torch::tensor({ 200 }, opts); // Silk Polyester

	// General Inter Yarn Parameters
	Tensor kc = torch::tensor({ 1.0 }, opts);
	Tensor mu = torch::tensor({ 0.5 }, opts);
	Tensor df = torch::tensor({ 1000.0 }, opts);
	Tensor kf = torch::tensor({ 1000.0 }, opts);

	std::vector<Yarn> cotton_poly{ cotton_yarn, poly_yarn };  // Yarn (1,2)
	std::vector<Yarn> cotton_silk{ cotton_yarn, silk_yarn };  // Yarn (1,3) 
	std::vector<Yarn> poly_silk{ poly_yarn, silk_yarn }; // Yarn (2,3)

	Cloth cloth(width, length, height, L, R, cotton_poly, kc, kf, df, mu, S1, handle_stiffness, WovenPattern::Plain, InitPose::Upright);

	Environment env{ &cloth, G, wind };

	PhysicsCloth phy{ &env };
	phy.InitMV();

	Tensor h = torch::tensor({ 0.001 }, opts);

	std::vector<std::pair<int, int>> handles;

	//Define handle nodes
	handles.push_back(std::make_pair(0, 0));
	// The following line should be set to
	// Simulation 5 x 5 cloth : handles.push_back(std::make_pair(4, 0));
	// Simulation 10 x 10 cloth : handles.push_back(std::make_pair(9, 0));
	// Simulation 17 x 17 cloth : handles.push_back(std::make_pair(16, 0));
	// Simulation 25 x 25 cloth : handles.push_back(std::make_pair(24, 0));
	handles.push_back(std::make_pair(9, 0));

	cloth.set_handles(handles);

	//cloth.ResetMV_Para();
	//cloth.UpdataMV_Para_Heter_Yarn();

	std::string h5path{ "./data/cotton_poly_plain_wind_10_env.h5" };
	std::shared_ptr<H5File> file = std::make_shared<H5File>(h5path, H5F_ACC_TRUNC);

	int steps{ 100 };  // Set Simulation steps

	BlockProgressBar bar{
		option::BarWidth{50},
		option::Start{"["},
		option::End{"]"},
		option::ForegroundColor{Color::white},
		indicators::option::PostfixText{"Simulation Step: 0"},
		option::FontStyles{std::vector<FontStyle>{FontStyle::bold}} };
	float bar_step{ 100.0f / steps };
	float bar_progress{ 0.0f };

	for (int i = 0; i < steps; ++i) {
		std::string ind{ "Simulation Step: " + std::to_string(i + 1) };
		std::string objname = boost::str(boost::format("Cloth%|04|") % i);
		std::string save_path = "./objs/" + objname + ".obj";

		phy.FillMV();

		SaveClothHomoYarn(cloth, env, file, steps, i);

		const Tensor A{ phy.get_SpaM() - (phy.get_SpaFPos()) * torch::pow(h, 2) - phy.get_SpaFVel() * h };
		const Tensor b{ h * (phy.get_SpaF() - torch::mm(phy.get_SpaFVel().transpose(0,1), phy.get_SpaVel())) + torch::mm(phy.get_SpaM(), phy.get_SpaVel()) };

		Tensor new_vel= EigenSolver::apply(A, b);

		cloth.update(new_vel, phy.F_E_is_Slides[0], h);

		phy.ResetMV();

		RenderCloth(cloth, save_path, RenderType::MeshTriangle);

		bar_progress += bar_step;
		bar.set_option(option::PostfixText{ ind });
		bar.set_progress(bar_progress);
		
	}
	file->close();
}
