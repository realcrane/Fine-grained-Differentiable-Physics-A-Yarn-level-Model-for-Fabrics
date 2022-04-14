#pragma once
#include "Cloth.h"
#include "Obstacle.h"
#include "Wind.h"

struct Environment
{
	Cloth* cloth;

	Obstacle* table,* ground;

	Tensor G;
	Wind wind;

	Environment() : cloth{ nullptr }, table{ nullptr }, ground{nullptr} {}

	Environment(Cloth* _cloth, Tensor _G, Wind _wind) : 
		cloth{ _cloth }, table{ nullptr }, ground{nullptr}, G{ _G }, wind{ _wind }{}

	Environment(Cloth* _cloth, Obstacle* _table, Obstacle* _ground, Tensor _G, Wind _wind):
		cloth{ _cloth }, table{ _table }, ground{ _ground }, G{ _G }, wind{ _wind }{}
};
