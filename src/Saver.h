#pragma once
#include "PhysicsCloth.h"
#include "H5Cpp.h"

using namespace H5;

struct H5ClothSize {
	int width;
	int length;

	H5ClothSize() = default;
	H5ClothSize(int width, int length) : width{ width }, length{ length }{};
	H5ClothSize(const H5ClothSize& other) :
		width{ other.width }, length{ other.length }{};
};

struct H5Physical {
	double gravity;
	double rho;
	double R;
	double L;
	double Y;
	double B;
	double kc;
	double kf;
	double df;
	double mu;
	double S;

	double handle_stiffness;
	double wind_velocity_x;
	double wind_velocity_y;
	double wind_velocity_z;
	double wind_density;
	double wind_drag;

	int woven_pattern;

	H5Physical() = default;
	H5Physical(double gravity, double rho, double R, double L, double Y, double B, double kc, double kf, double df, double mu, double S,
		double handle_stiffness, double wind_velocity_x, double wind_velocity_y, double wind_velocity_z, double wind_density, double wind_drag,
		int woven_pattern) :
		gravity{ gravity }, rho{ rho }, R{ R }, L{ L }, Y{ Y }, B{ B }, kc{ kc }, kf{ kf }, df{ df }, mu{ mu }, S{ S },
		handle_stiffness{ handle_stiffness }, wind_velocity_x(wind_velocity_x), wind_velocity_y(wind_velocity_y),
		wind_velocity_z(wind_velocity_z), wind_density(wind_density), wind_drag(wind_drag), woven_pattern(woven_pattern){};
	H5Physical(const H5Physical& other) :
		gravity{ other.gravity }, rho{ other.rho }, R{ other.R }, L{ other.L },
		Y{ other.Y }, B{ other.B }, kc{ other.kc }, kf{ other.kf }, df{ other.df },
		mu{ other.mu }, S{ other.S }, handle_stiffness{ other.handle_stiffness },
		wind_velocity_x(other.wind_velocity_x), wind_velocity_y(other.wind_velocity_y),
		wind_velocity_z(other.wind_velocity_z), wind_density(other.wind_density), wind_drag(other.wind_drag),
		woven_pattern(other.woven_pattern){};
};

struct H5PhysicalHeterYarn {
	double gravity;
	double rho1;
	double rho2;
	double R;
	double L;
	double Y1;
	double Y2;
	double B1;
	double B2;
	double kc;
	double kf;
	double df;
	double mu;
	double S;

	double handle_stiffness;
	double wind_velocity_x;
	double wind_velocity_y;
	double wind_velocity_z;
	double wind_density;
	double wind_drag;

	int woven_pattern;

	H5PhysicalHeterYarn() = default;
	H5PhysicalHeterYarn(double gravity, double rho1, double rho2, double R, double L, double Y1, double Y2, double B1, double B2, double kc, double kf, double df, double mu, double S,
		double handle_stiffness, double wind_velocity_x, double wind_velocity_y, double wind_velocity_z, double wind_density, double wind_drag, int woven_pattern) :
		gravity{ gravity }, rho1{ rho1 }, rho2{ rho2 }, R{ R }, L{ L }, Y1{ Y1 }, Y2{ Y2 }, B1{ B1 }, B2{ B2 }, kc{ kc }, kf{ kf }, df{ df }, mu{ mu }, S{ S },
		handle_stiffness{ handle_stiffness }, wind_velocity_x(wind_velocity_x), wind_velocity_y(wind_velocity_y),
		wind_velocity_z(wind_velocity_z), wind_density(wind_density), wind_drag(wind_drag), woven_pattern(woven_pattern){};
	H5PhysicalHeterYarn(const H5PhysicalHeterYarn& other) :
		gravity{ other.gravity }, rho1{ other.rho1 }, rho2{ other.rho2 }, R{ other.R }, L{ other.L },
		Y1{ other.Y1 }, Y2{ other.Y2 }, B1{ other.B1 }, B2{ other.B2 }, kc{ other.kc }, kf{ other.kf }, df{ other.df },
		mu{ other.mu }, S{ other.S }, handle_stiffness{ other.handle_stiffness },
		wind_velocity_x(other.wind_velocity_x), wind_velocity_y(other.wind_velocity_y),
		wind_velocity_z(other.wind_velocity_z), wind_density(other.wind_density), wind_drag(other.wind_drag), woven_pattern(other.woven_pattern){};
};

void SaveTrainProc(H5File& file, const int& curr_step, const int& train_steps, const Tensor& loss, const torch::OrderedDict<std::string, Tensor>& paras);

void SaveTrainProc(H5File& file, const int& curr_step, const int& train_steps, const Tensor& loss, const std::map<std::string, Tensor>& paras);

void SaveClothHomoYarn(const Cloth& cloth, const Environment& env, std::shared_ptr<H5File> file, int step_num, int step);

void SaveClothHeterYarn(const Cloth& cloth, const Environment& env, std::shared_ptr<H5File> file, int step_num, int step);
