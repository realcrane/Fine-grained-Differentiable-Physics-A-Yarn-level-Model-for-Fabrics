#pragma once
#include <random>

class RandInt {
	std::default_random_engine re;
	std::uniform_int_distribution<> dist;
public:
	RandInt(int low, int high) :dist{ low, high } {}
	int operator()() { return dist(re); }
	void seed(int s) { re.seed(s); }
};