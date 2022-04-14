#pragma once
#include "Loader.h"

enum class RenderType {
	MeshTriangle,
	MeshRectangle,
	YarnSpline
};

void RenderCloth(Cloth& cloth, std::string save_path, RenderType type);