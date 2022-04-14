#pragma once
#include "Yarn.h"
#include "Mesh.h"

struct Face;

struct Node {
	// Crossing node
	Node() = default;
	Node(int _u, int _v, Tensor _LPos, Tensor _EPos, Tensor _LVel, Tensor _EVel, Tensor _LPosFix, Tensor _EPosBar, bool _is_edge, bool _flip_norm) :
		idx_u(_u), idx_v(_v), LPos(_LPos), EPos(_EPos), LVel(_LVel), EVel(_EVel), LPosFix(_LPosFix), EPosBar(_EPosBar), is_edge(_is_edge), flip_norm(_flip_norm) {};

	int idx_u, idx_v;
	Tensor LPos;
	Tensor EPos;
	Tensor LVel;
	Tensor EVel;
	Tensor EPosBar;
	Tensor LPosFix;
	Tensor n;
	Tensor comp;
	bool is_edge;
	bool is_fixed{ false };
	bool flip_norm{ false };

	std::set<Face*> adj_faces;  // node's adjacent faces
};

struct NodeForce {
	NodeForce() = default;
	NodeForce(int _u, int _v) : idx_u(_u), idx_v(_v) {
		Inertia_L = ZERO31;
		Inertia_E = ZERO21;
		MDotV_L = ZERO31;
		MDotV_E = ZERO21;
		Gravity_L = ZERO31;
		Gravity_E = ZERO21;
		Stretch_L = ZERO31;
		Stretch_E = ZERO21;
		Bending_L = ZERO31;
		Bending_E = ZERO21;
		Bending_Crimp_L = ZERO31;
		Friction_E = ZERO21;
		Shearing_L = ZERO31;
		ParaColl_E = ZERO21;
		Constrain_L = ZERO31;
		F_L = ZERO31;
		F_E = ZERO21;
	}

	void Reset() {
		Inertia_L = ZERO31;
		Inertia_E = ZERO21;
		MDotV_L = ZERO31;
		MDotV_E = ZERO21;
		Gravity_L = ZERO31;
		Gravity_E = ZERO21;
		Stretch_L = ZERO31;
		Stretch_E = ZERO21;
		Bending_L = ZERO31;
		Bending_E = ZERO21;
		Bending_Crimp_L = ZERO31;
		Friction_E = ZERO21;
		Shearing_L = ZERO31;
		ParaColl_E = ZERO21;
		Constrain_L = ZERO31;
		F_L = ZERO31;
		F_E = ZERO21;
	}

	int idx_u, idx_v;
	Tensor Inertia_L, Inertia_E;
	Tensor MDotV_L, MDotV_E;
	Tensor Gravity_L, Gravity_E;
	Tensor Stretch_L, Stretch_E;
	Tensor Bending_L, Bending_E;
	Tensor Bending_Crimp_L;
	Tensor Friction_E;
	Tensor Shearing_L;
	Tensor ParaColl_E;
	Tensor Constrain_L;
	Tensor F_L, F_E;
};

struct Edge {
	// Segment between to crossing node
	Edge() = default;
	Edge(Node* _n0, Node* _n1, std::string e_type) :
		n0(_n0), n1(_n1), edge_type(e_type) {
		if (edge_type == "warp") {
			delta_uv = n1->EPos[0] - n0->EPos[0];
			velocity_vec = torch::cat(
				{ torch::cat({n0->LVel, n0->EVel.index({0}).view({1,1})},0),
				torch::cat({n1->LVel, n1->EVel.index({0}).view({1,1})}, 0) }, 0
			);
		}
		else {
			delta_uv = n1->EPos[1] - n0->EPos[1];
			velocity_vec = torch::cat(
				{ torch::cat({n0->LVel, n0->EVel.index({1}).view({1,1})},0),
				torch::cat({n1->LVel, n1->EVel.index({1}).view({1,1})}, 0) }, 0
			);
		}

		w = (n1->LPos - n0->LPos) / delta_uv;
		l = torch::norm(n1->LPos - n0->LPos);
		d = (n1->LPos - n0->LPos) / l;
		P = EYE3 - torch::ger(d.squeeze(), d.squeeze());
	}

	Edge(Node* _n0, Node* _n1, Yarn yarn, std::string e_type) :
		n0(_n0), n1(_n1), yarn(yarn), edge_type(e_type) {
		if (edge_type == "warp") {
			delta_uv = n1->EPos[0] - n0->EPos[0];
			velocity_vec = torch::cat(
				{ torch::cat({n0->LVel, n0->EVel.index({0}).view({1,1})},0),
				torch::cat({n1->LVel, n1->EVel.index({0}).view({1,1})}, 0) }, 0
			);
		}
		else {
			delta_uv = n1->EPos[1] - n0->EPos[1];
			velocity_vec = torch::cat(
				{ torch::cat({n0->LVel, n0->EVel.index({1}).view({1,1})},0),
				torch::cat({n1->LVel, n1->EVel.index({1}).view({1,1})}, 0) }, 0
			);
		}

		w = (n1->LPos - n0->LPos) / delta_uv;
		l = torch::norm(n1->LPos - n0->LPos);
		d = (n1->LPos - n0->LPos) / l;
		P = EYE3 - torch::ger(d.squeeze(), d.squeeze());
	}

	void update() {
		if (edge_type == "warp") {
			delta_uv = n1->EPos[0] - n0->EPos[0];
			velocity_vec = torch::cat(
				{ torch::cat({n0->LVel, n0->EVel.index({0}).view({1,1})},0),
				torch::cat({n1->LVel, n1->EVel.index({0}).view({1,1})}, 0) }, 0
			);
		}
		else {
			delta_uv = n1->EPos[1] - n0->EPos[1];
			velocity_vec = torch::cat(
				{ torch::cat({n0->LVel, n0->EVel.index({1}).view({1,1})},0),
				torch::cat({n1->LVel, n1->EVel.index({1}).view({1,1})}, 0) }, 0
			);
		}

		w = (n1->LPos - n0->LPos) / delta_uv;
		l = torch::norm(n1->LPos - n0->LPos);
		d = (n1->LPos - n0->LPos) / l;
		P = EYE3 - torch::ger(d.squeeze(), d.squeeze());
	}

	Node* n0, * n1;
	Yarn yarn; // Yarn Material of this segment
	std::string edge_type;
	Tensor delta_uv, velocity_vec, w, d, l, P;
};

struct Bend_Seg {
	// two adjacent edges including three nodes
	Bend_Seg() = default;
	Bend_Seg(Node* _n2, Node* _n0, Node* _n1, std::string s_type):
		n2(_n2), n0(_n0), n1(_n1), seg_type(s_type) {
		if (s_type == "warp")
			delta_uv = n1->EPos[0] - n2->EPos[0];
		else
			delta_uv = n1->EPos[1] - n2->EPos[1];

		l1 = torch::norm(n1->LPos - n0->LPos);
		l2 = torch::norm(n2->LPos - n0->LPos);
		d1 = (n1->LPos - n0->LPos) / l1;
		d2 = (n2->LPos - n0->LPos) / l2;
		P1 = EYE3 - torch::ger(d1.squeeze(), d1.squeeze());
		P2 = EYE3 - torch::ger(d2.squeeze(), d2.squeeze());
		theta = torch::acos(torch::clamp(-torch::mm(d1.transpose(1, 0), d2), -1.0 + EPS, 1.0 - EPS));
	}

	Bend_Seg(Node* _n2, Node* _n0, Node* _n1, Yarn _yarn, std::string s_type) :
		n2(_n2), n0(_n0), n1(_n1), yarn(_yarn), seg_type(s_type) {
		if (s_type == "warp")
			delta_uv = n1->EPos[0] - n2->EPos[0];
		else
			delta_uv = n1->EPos[1] - n2->EPos[1];

		l1 = torch::norm(n1->LPos - n0->LPos);
		l2 = torch::norm(n2->LPos - n0->LPos);
		d1 = (n1->LPos - n0->LPos) / l1;
		d2 = (n2->LPos - n0->LPos) / l2;
		P1 = EYE3 - torch::ger(d1.squeeze(), d1.squeeze());
		P2 = EYE3 - torch::ger(d2.squeeze(), d2.squeeze());
		theta = torch::acos(torch::clamp(-torch::mm(d1.transpose(1, 0), d2), -1.0 + EPS, 1.0 - EPS));
	}

	void update() {
		if (seg_type == "warp")
			delta_uv = n1->EPos[0] - n2->EPos[0];
		else
			delta_uv = n1->EPos[1] - n2->EPos[1];

		l1 = torch::norm(n1->LPos - n0->LPos);
		l2 = torch::norm(n2->LPos - n0->LPos);
		d1 = (n1->LPos - n0->LPos) / l1;
		d2 = (n2->LPos - n0->LPos) / l2;
		P1 = EYE3 - torch::ger(d1.squeeze(), d1.squeeze());
		P2 = EYE3 - torch::ger(d2.squeeze(), d2.squeeze());
		theta = torch::acos(torch::clamp(-torch::mm(d1.transpose(1, 0), d2), -1.0 + EPS, 1.0 - EPS));
	}

	Node* n2, * n0, * n1;
	Yarn yarn;
	std::string seg_type;
	Tensor delta_uv;
	Tensor l1, l2;
	Tensor d1, d2;
	Tensor P1, P2;
	Tensor theta;
};

struct Shear_Seg {
	Shear_Seg() = default;
	Shear_Seg(Node* node, Node* node_warp, Node* node_weft) :
		n0(node), n1(node_warp), n3(node_weft) {
		if (node->flip_norm != node_warp->flip_norm &&
			node->flip_norm != node_weft->flip_norm)
			is_jamming = true;
		else
			is_jamming = false;
		l1 = torch::norm(n1->LPos - n0->LPos);
		l3 = torch::norm(n3->LPos - n0->LPos);
		d1 = ((n1->LPos - n0->LPos) / l1);
		d3 = ((n3->LPos - n0->LPos) / l3);
		P1 = EYE3 - torch::ger(d1.squeeze(), d1.squeeze());
		P3 = EYE3 - torch::ger(d3.squeeze(), d3.squeeze());
		phi = torch::acos(torch::mm(d1.transpose(1, 0), d3));
	}

	void update() {
		l1 = torch::norm(n1->LPos - n0->LPos);
		l3 = torch::norm(n3->LPos - n0->LPos);
		d1 = ((n1->LPos - n0->LPos) / l1);
		d3 = ((n3->LPos - n0->LPos) / l3);
		P1 = EYE3 - torch::ger(d1.squeeze(), d1.squeeze());
		P3 = EYE3 - torch::ger(d3.squeeze(), d3.squeeze());
		phi = torch::acos(torch::mm(d1.transpose(1, 0), d3));
	}

	Node* n0, * n1, * n3;
	Tensor l1, l3;
	Tensor d1, d3;
	Tensor P1, P3;
	Tensor phi;
	bool is_jamming;
};

struct Face {
	Face() = default;
	Face(Node* _n1, Node* _n2, Node* _n3) :
		n1(_n1), n2(_n2), n3(_n3) {
		area = 0.5 * torch::norm(torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos));
		n = torch::norm(torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos)).item<double>() == 0.0 ?
			torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos) :
			torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos) / torch::norm(torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos));
	}

	void update() {
		area = 0.5 * torch::norm(torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos));
		n = torch::norm(torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos)).item<double>() == 0.0 ?
			torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos) :
			torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos) / torch::norm(torch::cross(n2->LPos - n1->LPos, n3->LPos - n1->LPos));
	}

	bool is_include(Node* node) {
		// If face includes the given node.
		return (node == n1 || node == n2 || node == n3);
	}

	Node* n1, * n2, * n3;
	Tensor area;
	Tensor n;
};
