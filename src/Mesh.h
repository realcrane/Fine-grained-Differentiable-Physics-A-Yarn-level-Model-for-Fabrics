#pragma once
#include "Solvers.h"
#include "Constants.h"
#include "Geometry.h"

struct MeshVert;
struct MeshNode;
struct MeshEdge;
struct MeshFace;

struct MeshVert {

	int index; // Index of meshvert in Mesh.verts

	Tensor uv; // material space position

	Tensor a; // Vertex area

	MeshNode* node; // One vert must corresponde to a node

	std::vector<MeshFace*> adj_faces; // The faces including this vert

	MeshVert() = default;
	MeshVert(const Tensor& _uv) : index{ 0 }, uv{ _uv } {
		a = ZERO1;
	}

	MeshVert(int idx, const Tensor& _uv) : index{ idx }, uv{ _uv } {
		a = ZERO1;
	}
};

struct MeshNode {

	int index; // Index of meshvert in Mesh.nodes

	bool is_fixed; // Is Handle

	Tensor x, x_prev, x_old; // Cartesian Position, or world space position

	Tensor v; // Node's Velocity

	Tensor m; // Node mass

	Tensor a; // Node's area

	Tensor n; // Node normal

	std::vector<MeshVert*> verts; // One node may include multiple verts

	std::vector<MeshEdge*> adj_edges; // The edges including this node

	MeshNode() = default;

	MeshNode(const Tensor& x, const Tensor& v) :
		index{ 0 }, is_fixed{false}, x{ x }, x_prev{ x }, v{ v } {
		a = ZERO1;
		n = ZERO31;
	}

	MeshNode(const Tensor& x) :
		index{ 0 }, is_fixed{ false }, x{ x }, x_prev{ x }, v(torch::zeros({ 3,1 }, opts)) {
		a = ZERO1;
		n = ZERO31;
	}

	MeshNode(int idx, const Tensor& x) :
		index{ idx }, is_fixed{ false }, x{ x }, x_prev{ x }, v(torch::zeros({ 3,1 }, opts)) {
		a = ZERO1;
		n = ZERO31;
	}
};

struct MeshEdge {

	int index; // Index of meshvert in Mesh.edges

	MeshNode* n0, * n1; // One egde must be consist of two MeshNode's

	std::array<MeshFace*, 2> adj_faces;

	MeshEdge() = default;
	MeshEdge(MeshNode* n0, MeshNode* n1):
		index{ 0 }, n0{ n0 }, n1{ n1 }{}
};

struct MeshFace {

	int index; // Index of meshvert in Mesh.faces

	Tensor a;	// Face Area

	Tensor n; // Face normal

	Tensor m; // Face mass

	std::array<MeshVert*, 3> v; // One Face must be consist of three MeshVerts
	std::array<MeshEdge*, 3> adj_edges;

	MeshFace() = default;
	MeshFace(MeshVert* v0, MeshVert* v1, MeshVert* v2): index{ 0 }{
		v[0] = v0; 
		v[1] = v1;
		v[2] = v2;
	}
};

struct Mesh {
	bool isCloth; // Cloth or Obstacle

	std::vector<MeshVert> verts;
	std::vector<MeshNode> nodes;
	std::vector<MeshEdge> edges;
	std::vector<MeshFace> faces;

	Mesh() = default;

	void add_edges_if_needed(const MeshFace& face);

	void add(MeshVert vert);
	void add(MeshNode node);
	void add(MeshEdge edge);
	void add(MeshFace face);

	void add_adjecent();

	MeshEdge* get_edge (const MeshNode* n0, const MeshNode* n1);
	Tensor get_nodes_pos();

	void compute_ms_data();
	void compute_ws_data();

	void update_x_prev();
};

void connect(MeshVert* vert, MeshNode* node); // connect MeshVert to MeshNode

void compute_ms_vert(MeshVert& vert);

void compute_ms_node(MeshNode& node);

void compute_ms_edge(MeshEdge& edge);

void compute_ms_face(MeshFace& face);

void compute_ws_vert(MeshVert& vert);

void compute_ws_node(MeshNode& node);

void compute_ws_edge(MeshEdge& edge);

void compute_ws_face(MeshFace& face);