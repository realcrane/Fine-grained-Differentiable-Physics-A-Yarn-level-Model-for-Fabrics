#pragma once
#include "Mesh.h"

struct Obstacle
{
	Mesh obsMesh;
	Tensor density;

	void compute_mesh_mass();
};