#include "TrainHelper.h"

Tensor constraint_para(Tensor para, double upper, double lower) {
	//assert(lower > upper, "Upper limit should be greater than the lower one");
	double p_range = upper - lower;
	return p_range * torch::sigmoid(para) + lower;
}

double invert_constraint_para(double para, double upper, double lower) {
	//assert(lower > upper, "Upper limit should be greater than the lower one");
	double p_range = upper - lower;
	return -std::log(p_range / (para - lower) - 1.0);
}

bool SimStep(Cloth& cloth, PhysicsCloth& sim_phy, const Tensor& h)
{
	sim_phy.FillMV();

	const Tensor A{ sim_phy.get_SpaM() - (sim_phy.get_SpaFPos()) * torch::pow(h, 2) - sim_phy.get_SpaFVel() * h };
	const Tensor b{ h * (sim_phy.get_SpaF() - torch::mm(sim_phy.get_SpaFVel().transpose(0,1), sim_phy.get_SpaVel())) + torch::mm(sim_phy.get_SpaM(), sim_phy.get_SpaVel()) };

	if (DetectNaN(A, false)) {
		std::cout << "NaN in Matrix A";
		return false;
	}
	if (DetectNaN(b, false)) {
		std::cout << "NaN in Vector b";
		return false;
	}
	if (DetectInf(A, false)) {
		std::cout << "Inf in Matrix A";
		return false;
	}
	if (DetectInf(b, false)) {
		std::cout << "Inf in Vector b";
		return false;
	}

	Tensor new_vel = EigenSolver::apply(A, b);

	if (DetectNaN(new_vel, false)) {
		std::cout << "NaN in Vector new velocity";
		return false;
	}
	if (DetectInf(new_vel, false)) {
		std::cout << "Inf in Vector new velocity";
		return false;
	}

	cloth.update(new_vel, sim_phy.F_E_is_Slides[0], h);
	sim_phy.ResetMV();

	return true;
}

bool SimStep(Cloth& cloth, PhysicsCloth& sim_phy, const Tensor& h, const Tensor& ext_force)
{

	sim_phy.FillMV();

	const Tensor A{ sim_phy.get_SpaM() - (sim_phy.get_SpaFPos()) * torch::pow(h, 2) - sim_phy.get_SpaFVel() * h };

	const Tensor b{ h * (sim_phy.get_SpaF() + ext_force - torch::mm(sim_phy.get_SpaFVel().transpose(0,1), sim_phy.get_SpaVel())) + torch::mm(sim_phy.get_SpaM(), sim_phy.get_SpaVel()) };

	if (DetectNaN(A, false)) {
		std::cout << "NaN in Matrix A";
		return false;
	}
	if (DetectNaN(b, false)) {
		std::cout << "NaN in Vector b";
		return false;
	}
	if (DetectInf(A, false)) {
		std::cout << "Inf in Matrix A";
		return false;
	}
	if (DetectInf(b, false)) {
		std::cout << "Inf in Vector b";
		return false;
	}

	Tensor new_vel = EigenSolver::apply(A, b);

	if (DetectNaN(new_vel, false)) {
		std::cout << "NaN in Vector new velocity";
		return false;
	}
	if (DetectInf(new_vel, false)) {
		std::cout << "Inf in Vector new velocity";
		return false;
	}

	cloth.update(new_vel, sim_phy.F_E_is_Slides[0], h);
	sim_phy.ResetMV();

	return true;
}

bool SimStep(Cloth& cloth, PhysicsCloth& phy_sim, Collision& collision, Tensor& h, bool is_para)
{
	phy_sim.FillMV();

	const Tensor A{ phy_sim.get_SpaM() - (phy_sim.get_SpaFPos()) * torch::pow(h, 2) - phy_sim.get_SpaFVel() * h };
	const Tensor b{ h * (phy_sim.get_SpaF() - torch::mm(phy_sim.get_SpaFVel().transpose(0,1), phy_sim.get_SpaVel())) + torch::mm(phy_sim.get_SpaM(), phy_sim.get_SpaVel()) };

	if (DetectNaN(A, false)) {
		std::cout << "NaN in Matrix A";
		return false;
	}
	if (DetectNaN(b, false)) {
		std::cout << "NaN in Vector b";
		return false;
	}
	if (DetectInf(A, false)) {
		std::cout << "Inf in Matrix A";
		return false;
	}
	if (DetectInf(b, false)) {
		std::cout << "Inf in Vector b";
		return false;
	}

	Tensor new_vel = EigenSolver::apply(A, b);

	if (DetectNaN(new_vel, false)) {
		std::cout << "NaN in Vector new velocity";
		return false;
	}
	if (DetectInf(new_vel, false)) {
		std::cout << "Inf in Vector new velocity";
		return false;
	}

	cloth.update(new_vel, phy_sim.F_E_is_Slides[0], h);
	phy_sim.ResetMV();

	cloth.clothMesh.compute_ms_data();
	cloth.clothMesh.compute_ws_data();

	bool changed = collision.collision_response(0.1);

	if (changed) {
		cloth.update();
		cloth.clothMesh.compute_ms_data();
		cloth.clothMesh.compute_ws_data();
	}

	return true;
}

