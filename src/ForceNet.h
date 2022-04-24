#pragma once
#include <torch/torch.h>

struct ForceNetImpl : torch::nn::Module {

    ForceNetImpl(int64_t input, int64_t output) :
        fc1(register_module("fc1", torch::nn::Linear(input, 125))),
        fc2(register_module("fc2", torch::nn::Linear(125, 60))),
        fc3(register_module("fc3", torch::nn::Linear(60, output)))
    {

    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = 0.004 * torch::sigmoid(fc3(x));

        return x;
    }

    torch::nn::Linear fc1, fc2, fc3;

};

TORCH_MODULE(ForceNet);
