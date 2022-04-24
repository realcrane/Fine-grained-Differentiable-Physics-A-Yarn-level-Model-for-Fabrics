#pragma once
#include "Render.h"
#include "Obstacle.h"
#include "Collision.h"

Tensor constraint_para(Tensor para, double upper = 1.0, double lower = 0.0);

double invert_constraint_para(double para, double upper = 1.0, double lower = 0.0);

bool SimStep(Cloth& cloth, PhysicsCloth& sim_phy, const Tensor& h);

bool SimStep(Cloth& cloth, PhysicsCloth& sim_phy, const Tensor& h, const Tensor& ext_force);

bool SimStep(Cloth& cloth, PhysicsCloth& phy_sim, Collision& collision, Tensor& h, bool is_para);