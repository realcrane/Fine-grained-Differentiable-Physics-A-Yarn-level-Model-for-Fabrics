#pragma once
#include <regex>
#include "Saver.h"

std::pair<H5ClothSize, H5Physical> cloth_property_loader(std::string filepath);

std::pair<H5ClothSize, H5PhysicalHeterYarn> cloth_property_loader_heter(std::string filepath);

ClothState data_loader(std::string filepath, int start, int end, torch::TensorOptions opt);

void load_obj(std::string filepath, Mesh& mesh);

void load_obj_train(const std::string& filepath, Tensor& GTPos);
