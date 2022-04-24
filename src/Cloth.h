#include <iostream>
#include <vector>
#include <execution>
#include <map>

#include "ClothUnit.h"
#include "Debugger.h"

using namespace torch::indexing;

enum class InitPose {
	Upright,
	Flat
};

enum class WovenPattern {
	Plain,
	Twill,
	Satin
};

struct ClothState {
	Tensor LPos;
	Tensor LPosFix;
	Tensor LVel;
	Tensor EPos;
	Tensor EPosBar;
	Tensor EVel;
};

struct Cloth {
	int width, length;
	double hight, L, R;
	Tensor kf, df, mu, S, kc, handle_stiffness;

	std::vector<Yarn> yarns;

	double jamming_thrd;
	WovenPattern woven_pattern;
	int num_nodes, num_cros_nodes, num_LLD, num_ELD, num_EED;

	std::vector<Tensor> Zeros_L, Zeros_E;
	std::vector<Tensor> Zeros_LL, Zeros_EL, Zeros_LE, Zeros_EE;

	std::map<std::string, Node> nodes, warp_nodes, weft_nodes;
	std::vector<Edge> edges;
	std::vector<Bend_Seg> bend_segs, crimp_bend_segs;
	std::vector<Shear_Seg> shear_segs;
	std::vector<Face> faces;

	Mesh clothMesh;

	Cloth() = default;

	Cloth(int _width, int _length,
		double hight, double _L,
		double _R, std::vector<Yarn> _yarns,
		Tensor _kc, Tensor _kf, Tensor _df, Tensor _mu, Tensor _S,
		Tensor h_stiff, WovenPattern _woven_pattern,
		InitPose _init_pose);

	void build(InitPose init_pose);

	void update();
	void update(const Tensor & new_vel, const Tensor& F_E_is_Slide, const Tensor & h);

	void InitClothUnits();
	void UpdateClothUnits();

	void set_handles(const std::vector<std::pair<int, int>>& idxs);

	ClothState get_ClothState() const;

	void set_ClothState(const ClothState & state);
	void set_ClothStateMesh(const ClothState& state);	

	Tensor NodeNorm(const Node& n_cen, const Node& n_left, const Node& n_right, const Node& n_top, const Node& n_bot);

	void ConnectNodeFace();

	void MeshMass(int u, int v);
};


