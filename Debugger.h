#pragma once
#include <chrono>
#include "Constants.h"

inline bool DetectNaN(const Tensor& tested_tensor, bool is_select) {
	if (torch::isnan(tested_tensor).any().item<bool>()) {
		std::cout << "NaN Detected" << std::endl;
		if (is_select) {
			std::cout << "Do you want to print the tensor? [y/n]" << std::endl;
			char rep = std::cin.get();
			if (rep == 'y')
				std::cout << tested_tensor << std::endl;
		}
		return true;
	}
	return false;
}

inline bool DetectInf(const Tensor& tested_tensor, bool is_select) {
	if (torch::isinf(tested_tensor).any().item<bool>()) {
		std::cout << "Inf Detected" << std::endl;
		if (is_select) {
			std::cout << "Do you want to print the tensor? [y/n]" << std::endl;
			char rep = std::cin.get();
			if (rep == 'y')
				std::cout << tested_tensor << std::endl;
		}
		return true;
	}
	return false;
}

inline std::string Tensor_equal(const Tensor& a, const Tensor& b, int decimal = 0) {
	int precision{ static_cast<int>(pow(10 , decimal)) };
	return (torch::round(a * precision) * precision == torch::round(b * precision) * precision).all().item<bool>()?
		"Equal":"Unequal";
}

struct  StopWatch {
	StopWatch(std::chrono::nanoseconds& result):
		result {result},
		start{ std::chrono::high_resolution_clock::now() }{}
	~StopWatch() {
		result = std::chrono::high_resolution_clock::now() - start;
	}
private:
	std::chrono::nanoseconds& result;
	const std::chrono::time_point<std::chrono::high_resolution_clock> start;
};