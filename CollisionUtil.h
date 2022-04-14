#pragma once
#include "BVH.h"

struct AccelStruct {
	BVHTree tree;

	BVHNode* root;
	std::vector<BVHNode*> leaves;

	AccelStruct() = default;
	AccelStruct(Mesh& mesh, bool ccd);

	void update();
};

void collect_leaves(BVHNode* node, std::vector<BVHNode*>& leaves);

void mark_all_inactive(AccelStruct& acc);

void mark_active(AccelStruct& acc, const MeshFace* face);

void mark_descendants(BVHNode* node, bool active);

void mark_ancestors(BVHNode* node, bool active);

bool find_mesh(const MeshNode* node, const Mesh& mesh);

std::pair<bool, int> find_in_meshes(const MeshNode* node, const Mesh& cloth_mes, const Mesh& obs_mesh);