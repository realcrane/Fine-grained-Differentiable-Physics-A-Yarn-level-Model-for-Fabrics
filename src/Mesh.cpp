#include "Mesh.h"

void connect(MeshVert* vert, MeshNode* node) {

	vert->node = node; // assign node to vert

	if(std::find(node->verts.begin(), node->verts.end(), vert) == node->verts.end())
		node->verts.push_back(vert); // assign vert to vert;
}

MeshEdge* Mesh::get_edge(const MeshNode* n0, const MeshNode* n1) {
	for (int e = 0; e < n0->adj_edges.size(); ++e) {
		MeshEdge* edge = n0->adj_edges[e];
		if (edge->n0 == n1 || edge->n1 == n1)
			return edge;
	}
	return nullptr;
}

void Mesh::add_edges_if_needed(const MeshFace& face) {
	for (int idx_n0 = 0; idx_n0 < 3; ++idx_n0) {
		int idx_n1{ idx_n0 != 2 ? idx_n0 + 1 : 0 };
		if (get_edge(face.v[idx_n0]->node, face.v[idx_n1]->node) == nullptr) {
			this->add(MeshEdge(face.v[idx_n0]->node, face.v[idx_n1]->node));
			std::cout << face.v[idx_n0]->node->x << std::endl;
			std::cout << face.v[idx_n1]->node->x << std::endl;
			std::cout << "Add Edge" << std::endl;
		}
	}
}

void Mesh::add(MeshVert vert){
	vert.node = nullptr;
	vert.adj_faces.clear();
	vert.index = verts.size();
	verts.push_back(vert);
}

void Mesh::add(MeshNode node){
	node.index = nodes.size();
	node.adj_edges.clear();
	nodes.push_back(node);
}

void Mesh::add(MeshEdge edge){
	edge.adj_faces[0] = nullptr;
	edge.adj_faces[1] = nullptr;
	edge.index = edges.size();
	edges.push_back(edge);

	//if (std::find(edge.n0->adj_edges.begin(), edge.n0->adj_edges.end(), edges) == edge.n0->adj_edges.end())
	//	new_edge->n0->adj_edges.push_back(new_edge);

	//if (std::find(new_edge->n1->adj_edges.begin(), new_edge->n1->adj_edges.end(), new_edge) == edge.n1->adj_edges.end())
	//	new_edge->n1->adj_edges.push_back(new_edge);
}

void Mesh::add(MeshFace face){
	face.adj_edges[0] = nullptr;
	face.adj_edges[1] = nullptr;
	face.adj_edges[2] = nullptr;
	face.index = faces.size() ;
	faces.push_back(face);

	/*add_edges_if_needed(faces.back());*/

	//for (int idx_n0 = 0; idx_n0 < 3; ++idx_n0) {
	//	// add face's verts' adjcent faces.
	//	if (std::find(face.v[idx_n0]->adj_faces.begin(), face.v[idx_n0]->adj_faces.end(), &faces.back()) == face.v[idx_n0]->adj_faces.end())
	//		faces.back().v[idx_n0]->adj_faces.push_back(&faces.back());

	//	int idx_n1{ idx_n0 != 2 ? idx_n0 + 1 : 0 };
	//	MeshEdge* edge = get_edge(faces.back().v[idx_n0]->node, faces.back().v[idx_n1]->node);
	//	faces.back().adj_edges[idx_n0] = edge;
	//	int edge_side{ edge->n0 == faces.back().v[idx_n0]->node ? 0 : 1 };
	//	edge->adj_faces[edge_side] = &faces.back();
	//}
}

void Mesh::add_adjecent() {
	// Set nodes' adjecent edges
	for (int e = 0; e < edges.size(); ++e) {
		if (find(edges[e].n0->adj_edges.begin(), edges[e].n0->adj_edges.end(), &edges[e]) == edges[e].n0->adj_edges.end()) {
			edges[e].n0->adj_edges.push_back(&edges[e]);
		}
		if (find(edges[e].n1->adj_edges.begin(), edges[e].n1->adj_edges.end(), &edges[e]) == edges[e].n1->adj_edges.end()) {
			edges[e].n1->adj_edges.push_back(&edges[e]);
		}
	}

	//for (int f = 0; f < faces.size(); ++f) {
	//	add_edges_if_needed(faces[f]);
	//}

	// Set verts adjecent faces;
	for (int f = 0; f < faces.size(); ++f) {
		for (int v = 0; v < 3; ++v) {
			if (std::find(faces[f].v[v]->adj_faces.begin(), faces[f].v[v]->adj_faces.end(), &faces[f]) == faces[f].v[v]->adj_faces.end())
				faces[f].v[v]->adj_faces.push_back(&faces[f]);

			int idx_n1{ v != 2 ? v + 1 : 0 };
			MeshEdge* edge = get_edge(faces[f].v[v]->node, faces[f].v[idx_n1]->node);

			faces[f].adj_edges[v] = edge;
			//int edge_side{ edge->n0 == faces[f].v[v]->node ? 0 : 1 };
			//edge->adj_faces[edge_side] = &faces[f];
		}
	}
}

Tensor Mesh::get_nodes_pos() {
	int num_nodes{ static_cast<int>(nodes.size()) };
	Tensor xs{ torch::zeros({num_nodes, 3, 1}) };
	for (int i = 0; i < num_nodes; ++i) {
		xs[i] = nodes.at(i).x;
	}
	return xs;
}

void Mesh::compute_ms_data() {
	for (int f = 0; f < faces.size(); ++f) 
		compute_ms_face(faces[f]);
	//for (int e = 0; e < edges.size(); ++e)
	//	compute_ms_edge(edges[e]);
	for (int v = 0; v < verts.size(); ++v)
		compute_ms_vert(verts[v]);
	for (int n = 0; n < nodes.size(); ++n)
		compute_ms_node(nodes[n]);
}

void Mesh::compute_ws_data() {
	for (int f = 0; f < faces.size(); ++f) {
		compute_ws_face(faces[f]);
	}
	//if (isCloth) {
	//	for (int e = 0; e < edges.size(); ++e)
	//		compute_ws_edge(edges[e]);
	//}
	for (int n = 0; n < nodes.size(); ++n)
		compute_ws_node(nodes[n]);
}

void compute_ms_vert(MeshVert& vert) {
	for (int f = 0; f < vert.adj_faces.size(); ++f) {
		MeshFace* face = vert.adj_faces[f];
		vert.a = vert.a + face->a / 3.0;
	}
}

void compute_ms_node(MeshNode& node) {
	for (int v = 0; v < node.verts.size(); ++v) {
		node.a = node.a + node.verts[v]->a;
	}
}

void compute_ms_edge(MeshEdge& edge) {
	//std::cout << "Computer Mesh Edge" << std::endl;
}

void compute_ms_face(MeshFace& face) {
	//std::cout << "Computer Mesh Face" << std::endl;

	Tensor e1{ face.v[1]->uv - face.v[0]->uv };
	Tensor e2{ face.v[2]->uv - face.v[0]->uv };

	face.a = 0.5 * torch::det(torch::cat({ e1, e2 }, 1));
}

void compute_ws_node(MeshNode& node) {

	for (int v = 0; v < node.verts.size(); ++v) {
		const MeshVert* vert = node.verts[v];
		const std::vector<MeshFace*>& adjfs = vert->adj_faces;
		for (MeshFace* face : vert->adj_faces) {
			node.n = node.n + face->n;
		}
	}

	Tensor normal_length{ torch::norm(node.n) };
	if (normal_length.item<double>() == 0.0)
		node.n = node.n;
	else
		node.n = node.n / normal_length;
}

void compute_ws_edge(MeshEdge& edge) {
	//std::cout << "Compute World Space Edge" << std::endl;
}

void compute_ws_face(MeshFace& face) {
	//std::cout << "Compute World Space Face" << std::endl;

	Tensor x0 = face.v[0]->node->x;
	Tensor x1 = face.v[1]->node->x;
	Tensor x2 = face.v[2]->node->x;

	Tensor face_n{ torch::cross(x1 - x0, x2 - x0) };
	Tensor l{ torch::norm(face_n) };

	if (l.item<double>() == 0.0)
		face.n = face_n;
	else
		face.n = face_n / l;
}

void Mesh::update_x_prev() {
	for (int n = 0; n < nodes.size(); ++n)
		nodes[n].x_prev = nodes[n].x;
}