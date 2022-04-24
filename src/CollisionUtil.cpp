#include "CollisionUtil.h"

AccelStruct::AccelStruct(Mesh& mesh, bool ccd):
	tree( BVHTree(ccd, mesh)), 
	root (tree._root),
	leaves (std::vector<BVHNode*>(mesh.faces.size())) {
	if (root != nullptr)
		collect_leaves(root, leaves);
}

void AccelStruct::update() {
	// Update BVH Tree
	if (root != nullptr)
		tree.refit();
}

void collect_leaves(BVHNode* node, std::vector<BVHNode*>& leaves) {
	if (node->isLeaf()) {
		int f{ node->_face->index };
		if (f > leaves.size())
			leaves.resize(f + 1);
		leaves[f] = node;
	}
	else {
		collect_leaves(node->_left, leaves);
		collect_leaves(node->_right, leaves);
	}
}

void mark_all_inactive(AccelStruct& acc) {
	// Deactive Downward
	if (acc.root)
		mark_descendants(acc.root, false);
}

void mark_active(AccelStruct& acc, const MeshFace* face) {
	// Active Upward
	if (acc.root)
		mark_ancestors(acc.leaves[face->index], true);
}

void mark_descendants(BVHNode* node, bool active) {
	// Alter active state from root to leaf
	node->_actived = active;
	if (!node->isLeaf()) {
		mark_descendants(node->_left, active);
		mark_descendants(node->_right, active);
	}
}

void mark_ancestors(BVHNode* node, bool active) {
	// Alter active node from left to root
	node->_actived = active;
	if (!node->isRoot())
		mark_ancestors(node->_parent, active);
}

bool find_mesh(const MeshNode* node, const Mesh& mesh) {
	// Find node in mesh
	if (node->index < mesh.nodes.size() && node == &mesh.nodes.at(node->index))
		return true;
	else
		return false;
}

std::pair<bool, int> find_in_meshes(const MeshNode* node, const Mesh& cloth_mesh, const Mesh& obs_mesh) {
	// Find node in a cloth mesh and multiple obstable meshes
	if (find_mesh(node, cloth_mesh))
		return { true, -1 }; // Cloth
	else if (find_mesh(node, obs_mesh))
		return { false, 0 }; // Obstacle
	else
		return { false, 1 }; // Ground
}
