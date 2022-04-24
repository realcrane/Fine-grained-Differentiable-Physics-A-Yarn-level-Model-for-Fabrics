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
