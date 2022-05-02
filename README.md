# Fine-grained-Differentiable-Physics-A-Yarn-level-Model-for-Fabrics
Our ICLR 2022 paper: Fine-grained Differentiable Physics: A Yarn-level Model for Fabrics

## Compiler and Dependencies
* MSVC 19.29 or GCC 10.2.0
* [Alglib 3.18.0](https://www.alglib.net/)
* [Boost 1.7.5](https://www.boost.org/)
* [Eigen 3.3.9](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* [HDF5 1.10.8](https://www.hdfgroup.org/downloads/hdf5)
* [indicators](https://github.com/p-ranav/indicators)
* [Pytorch 1.9.0](https://github.com/pytorch/pytorch)

## Simulation
The functions for simulations include:
* SimHomo(): Simulate homogeneous cloth.
* SimHeter(): Simulate blend woven cloth.
* SimCollision(): A senario includes cloth-obstacle collision and cloth self-collision.

## Training
The functions for training include:
* TrainHeterFew(): Learn yarn's density, stretching stiffness, and bending stiffness.
* TrainHeterFull(): Learn yarn's density, stretching stiffness, bending stiffness, shearing stiffness, and sliding friction coefficient.
* TrainControl(): Learn the needed force to throw a piece of cloth in target place.

## Notes
1. The visual results are saved as a sequence of .obj files which can be viewed in Blender with [Stop Motion OBJ](https://github.com/neverhood311/Stop-motion-OBJ) plug-in.
2. The simulation data, i.e positions and velocity, and training data are saved in hdf5 files which can be viewed in [HDF View](https://www.hdfgroup.org/downloads/hdfview/).

## Authors
Authors
Deshan Gong, Zhanxing Zhu, Andy Bulpitt and He Wang

Deshan Gong, scdg@leeds.ac.uk

He Wang, h.e.wang@leeds.ac.uk, [Personal website](https://drhewang.com)

Project Webpage: http://drhewang.com/pages/diffcloth.html

## Citation (Bibtex)
Please cite our paper if you find it useful:

    @InProceedings{Gong_Fine_2022,
    author={Deshan Gong, Zhanxing Zhu, Andy Bulpitt and He Wang},
    booktitle={Proceedings of the International Confernece on Learning Representations},
    title={Fine-grained Differentiable Physics: A Yarn-level Model for Fabrics},
    year={2022}
}

## License

Copyright (c) 2022, The University of Leeds, UK.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
