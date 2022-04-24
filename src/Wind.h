#pragma once
#include "torch/torch.h"

using torch::Tensor;

struct Wind{
	Tensor WindVelocity;
	Tensor WindDensity;
	Tensor WindDrag;

	// Constructor
	Wind() = default;
	Wind(Tensor _wind_vel, Tensor wind_den, Tensor wind_drag) :
		WindVelocity{ _wind_vel },
		WindDensity{ wind_den },
		WindDrag{ wind_drag }{};

	// Destructor
	~Wind() = default;

	// Copy Constructor
	Wind(const Wind& other) :
		WindVelocity{ other.WindVelocity },
		WindDensity{ other.WindDensity },
		WindDrag{ other.WindDrag }{};

	// Copy Assignment
	Wind& operator=(const Wind& other) {
		if (this == &other) return *this;
		WindVelocity = other.WindVelocity;
		WindDensity = other.WindDensity;
		WindDrag = other.WindDrag;
		return *this;
	}
};