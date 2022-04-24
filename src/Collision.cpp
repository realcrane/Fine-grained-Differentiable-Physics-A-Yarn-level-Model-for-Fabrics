#include "Collision.h"

Tensor arr2ten(real_2d_array a) {
	int n = a.rows(), m = a.cols();
	std::vector<double> tmp;
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			tmp.push_back(a[i][j]);
	Tensor ans = torch::tensor(tmp, opts).reshape({ n, m });
	return ans;
}

Tensor ptr2ten(int* a, int n) {
	std::vector<int> b;
	for (int i = 0; i < n; ++i)
		b.push_back(a[i]);
	return torch::tensor(b, torch::dtype(torch::kI32));
}

Tensor ptr2ten(double* a, int n) {
	std::vector<double> b;
	for (int i = 0; i < n; ++i)
		b.push_back(a[i]);
	return torch::tensor(b, opts);
}

real_2d_array ten2arr(Tensor a) {
	int n = a.size(0), m = a.size(1);
	auto foo_a = a.accessor<double, 2>();
	real_2d_array ans;
	ans.setlength(n, m);
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			ans[i][j] = foo_a[i][j];
	return ans;
}

real_1d_array ten1arr(Tensor a) {
	int n = a.size(0);
	auto foo_a = a.accessor<double, 1>();
	real_1d_array ans;
	ans.setlength(n);
	for (int i = 0; i < n; ++i)
		ans[i] = foo_a[i];
	return ans;
}

Tensor get_subvec(const double* x, int i) {
	return torch::tensor({ x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2] }, opts);
}

void set_subvec(double* x, int i, const Tensor& xi) {
	for (int j = 0; j < 3; ++j)
		x[i * 3 + j] = xi[j].item<double>();
}

inline bool is_in(const MeshNode* node, std::array<MeshNode*, 4> nodes) {
	for (int i = 0; i < nodes.size(); ++i) {
		if (node == nodes[i])
			return true;
	}
	return false;
}

inline bool is_in(const MeshNode* node, std::vector<MeshNode*> nodes) {
	for (int i = 0; i < nodes.size(); ++i) {
		if (node == nodes[i])
			return true;
	}
	return false;
}

bool Collision::collision_response(double damp_fac) {

	cloth_acc = new AccelStruct(*cloth_mesh, ccd);
	obs_acc = new AccelStruct(*obs_mesh, ccd);
	grd_acc = new AccelStruct(*ground_mesh, ccd);

	std::vector<ImpactZone*> zones, zones_prev;

	for (int c = 0; c < cloth_mesh->nodes.size(); ++c) {
		cloth_mesh->nodes[c].x_old = cloth_mesh->nodes[c].x;
	}

	bool changed{ false };
	int iter{ 0 };
	for (iter = 0; iter < max_iter; ++iter) {
		zones.clear();
		for (auto p : zones_prev) {
			zones.push_back(p);
		}
		if (!zones.empty()) {
			update_active(zones);
		}

		find_impacts();

		indenpendent_impacts();

		if (impacts.empty())
			break;
		else
			std::cout << "Collision Detected" << std::endl;
			
		// Add and Merge impacts into impact zones
		add_impacts(zones);

		for (int z = 0; z < zones.size(); z++) {
			if (zones[z]->active) {
				changed = true;
				apply_inelastic_projection(zones[z], thickness);
			}
		}
		cloth_acc->update();
		obs_acc->update();
		zones_prev = zones;
	}

	if (iter < max_iter) {
		std::cout << "Collision Converge" << std::endl;
	}
	else {
		std::cout << "Collision Coverge Failed" << std::endl;
		exit(1);
	}

	// updata velocity, world space, material space, and node previous position data.
	if (changed) {
		for (int n = 0; n < cloth_mesh->nodes.size(); ++n) {
			// Update Cloth Nodes Velocity
			cloth_mesh->nodes[n].v = cloth_mesh->nodes[n].v + damp_fac * (cloth_mesh->nodes[n].x - cloth_mesh->nodes[n].x_old)/ h;
		}
	}

	//for (int z = 0; z < zones.size(); ++z) {
	//	delete zones[z];
	//}

	delete cloth_acc;
	delete obs_acc;
	delete grd_acc;

	cloth_acc = nullptr;
	obs_acc = nullptr;
	grd_acc = nullptr;

	return changed;
}

bool Collision::is_free(const MeshNode* node) const {
	if (node->index < cloth_mesh->nodes.size() &&
		node == &cloth_mesh->nodes.at(node->index))
		return true;
	else
		return false;
}

void Collision::update_active(const std::vector<ImpactZone*>& zones) {
	// make BVHTree node active state false
	mark_all_inactive(*cloth_acc);
	mark_all_inactive(*obs_acc);
	mark_all_inactive(*grd_acc);

	for (int z = 0; z < zones.size(); ++z) {
		if (!zones[z]->active)
			continue;

		for (int n = 0; n < zones[z]->nodes.size(); ++n) {
			auto [is_in_cloth, idx_obs] = find_in_meshes(zones[z]->nodes[n], *cloth_mesh, *obs_mesh); // Find node in meshes
			AccelStruct* acc{ nullptr };
			if (is_in_cloth)
				acc = cloth_acc; // In Cloth
			else if (idx_obs == 0)
				acc = obs_acc; // In Table
			else if (idx_obs == 1)
				acc = grd_acc; // In Ground
			for (int v = 0; v < zones[z]->nodes[n]->verts.size(); ++v)
				for (int f = 0; f < zones[z]->nodes[n]->verts[v]->adj_faces.size(); ++f)
					mark_active(*acc, zones[z]->nodes[n]->verts[v]->adj_faces[f]);
		}
	}
}

void Collision::find_impacts() {
	impact_faces.clear();
	impacts.clear();

	for (int t = 0; t < NThrd; ++t) {
		impact_faces_para[t].clear();
		impacts_para[t].clear();
	}

	for_overlapping_faces();

	for (int t = 0; t < NThrd; ++t) {
		impact_faces.insert(impact_faces.end(), impact_faces_para[t].begin(), impact_faces_para[t].end());
	}

#pragma omp parallel for
	for (int i = 0; i < impact_faces.size(); ++i) {
		comp_face_impacts(impact_faces[i].first, impact_faces[i].second);
	}


	for (int t = 0; t < NThrd; ++t) {
		impacts.insert(impacts.end(), impacts_para[t].begin(), impacts_para[t].end());
	}
}

void Collision::comp_face_impacts(MeshFace* face0, MeshFace* face1) {

	int ID{ omp_get_thread_num() };
	std::array<kDOP18, 6> nodes_box, edges_box;
	std::array<kDOP18, 2> faces_box;
	Impact impact;
	// Fill node box
	for (int n = 0; n < 3; ++n) {
		nodes_box[n] = node_box(face0->v[n]->node, ccd);
		nodes_box[n + 3] = node_box(face1->v[n]->node, ccd);
	}
	// Fill edge box
	for (int n0_idx = 0; n0_idx < 3; ++n0_idx) {
		int n1_idx{ n0_idx != 2 ? n0_idx + 1 : 0 };
		edges_box[n0_idx] = nodes_box[n0_idx] + nodes_box[n1_idx];
		edges_box[n0_idx + 3] = nodes_box[n0_idx + 3] + nodes_box[n1_idx + 3];
	}
	// Fill face box
	faces_box[0] = nodes_box[0] + nodes_box[1] + nodes_box[2];
	faces_box[1] = nodes_box[3] + nodes_box[4] + nodes_box[5];

	 //Collision Test: nodes in face0 with face1
	for (int n = 0; n < 3; ++n) {
		if (!overlap(nodes_box[n], faces_box[1], thickness))
			continue;
		if (vf_collision_test(face0->v[n], face1, impact))
			impacts_para[ID].push_back(impact);
	}
	// Collision Test: nodes in face1 with face0
	for (int n = 0; n < 3; ++n) {
		if (!overlap(nodes_box[n + 3], faces_box[0], thickness))
			continue;
		if (vf_collision_test(face1->v[n], face0, impact))
			impacts_para[ID].push_back(impact);
	}
	// Collision Test: edges in face0 and edges in face1
	for (int e0 = 0; e0 < 3; ++e0)
		for (int e1 = 0; e1 < 3; ++e1) {
			if (!overlap(edges_box[e0], edges_box[e1 + 3], thickness))
				continue;
			if (ee_collision_test(face0->adj_edges[e0], face1->adj_edges[e1], impact))
				impacts_para[ID].push_back(impact);
		}
}

bool operator< (const Impact& impact0, const Impact& impact1) {
	return (impact0.t < impact1.t).item<int>();
}

void Collision::indenpendent_impacts() {
	std::vector<Impact> sorted = impacts;
	sort(sorted.begin(), sorted.end());
	std::vector<Impact> indep_impacts;	
	for (int imp = 0; imp < sorted.size(); ++imp) {
		const Impact& impact{ sorted[imp] };
		bool conf{ false };
		for (int ind_imp = 0; ind_imp < indep_impacts.size(); ++ind_imp) {
			if (conflict(impact, indep_impacts[ind_imp]))
				conf = true;
		}
		if (!conf)
			indep_impacts.push_back(impact);
	}
	this->impacts = indep_impacts;
}

bool Collision::conflict(const Impact& imp0, const Impact& imp1) {
	// May need to change: The code only check 
	return ((is_free(imp0.nodes[0]) && is_in(imp0.nodes[0], imp1.nodes))
		|| (is_free(imp0.nodes[1]) && is_in(imp0.nodes[1], imp1.nodes))
		|| (is_free(imp0.nodes[2]) && is_in(imp0.nodes[2], imp1.nodes))
		|| (is_free(imp0.nodes[3]) && is_in(imp0.nodes[3], imp1.nodes)));
}

void Collision::find_face_impacts(MeshFace* face0, MeshFace* face1) {
	int ID{omp_get_thread_num()};
	impact_faces_para[ID].push_back(std::make_pair(face0, face1));
}

void Collision::for_overlapping_faces() {
	int num_nodes_per_thread{NThrd};
	std::vector<BVHNode*> nodes;
	collect_upper_nodes(nodes, num_nodes_per_thread);

#pragma omp parallel for
	for (int n = 0; n < nodes.size(); ++n) {
		for_overlapping_faces(nodes[n]); // Collision between nodes that beneath this node
		for (int m = 0; m < n; ++m) {
			for_overlapping_faces(nodes[n], nodes[m]); // Collision between the node in this level
		}
		 for_overlapping_faces(nodes[n], obs_acc->root); // Collision between cloth and obstacle
		 for_overlapping_faces(nodes[n], grd_acc->root); // Collision between cloth and ground
	}
}

void Collision::for_overlapping_faces(BVHNode* node) {
	if (node->isLeaf() || !node->_actived)
		return;

	for_overlapping_faces(node->_left);
	for_overlapping_faces(node->_right);
	for_overlapping_faces(node->_left, node->_right);
}

void Collision::for_overlapping_faces(BVHNode* node0, BVHNode* node1) {
	if (!node0->_actived && !node1->_actived) // Jump over if nodes are not actived
		return;
	if (!overlap(node0->_box, node1->_box, thickness)) {
		// Jump over if nodes' boxes are not overlapped
		return;
	}
	if (node0->isLeaf() && node1->isLeaf()) {
		find_face_impacts(node0->_face, node1->_face);
	}
	else if (node0->isLeaf()) {
		for_overlapping_faces(node0, node1->_left);
		for_overlapping_faces(node0, node1->_right);
	}
	else {
		for_overlapping_faces(node0->_left, node1);
		for_overlapping_faces(node0->_right, node1);
	}
}

void Collision::collect_upper_nodes(std::vector<BVHNode*>& nodes, int num_nodes_per_thread) {

	nodes.push_back(cloth_acc->root);

	while (nodes.size() < num_nodes_per_thread) {
		std::vector<BVHNode*> children;
		for (int n = 0; n < nodes.size(); ++n) {
			// Go downward
			if (nodes[n]->isLeaf()) {
				children.push_back(nodes[n]);
			}
			else {
				children.push_back(nodes[n]->_left);
				children.push_back(nodes[n]->_right);
			}
		}
		if (children.size() == nodes.size()) {
			break;
		}
		nodes = children;
	}
}

bool vf_collision_test(const MeshVert* vert, const MeshFace* face, Impact& impact) {
	const MeshNode* node = vert->node;
	if (node == face->v[0]->node ||
		node == face->v[1]->node ||
		node == face->v[2]->node)
		return false; // Exclude the case where the face includes the vert
	return collision_test(ColliType::VF, node, face->v[0]->node, face->v[1]->node, face->v[2]->node, impact);
}

bool ee_collision_test(const MeshEdge* edge0, const MeshEdge* edge1, Impact& impact) {
	if (edge0->n0 == edge1->n0 || edge0->n0 == edge1->n1 ||
		edge0->n1 == edge1->n0 || edge0->n1 == edge1->n1)
		return false; // Exclude the case where two edges shares one same node
	return collision_test(ColliType::EE, edge0->n0, edge0->n1, edge1->n0, edge1->n1, impact);
}

bool collision_test(ColliType type, const MeshNode* node0, const MeshNode* node1, const MeshNode* node2, const MeshNode* node3, Impact& impact) {
	impact.type = type;
	impact.nodes[0] = const_cast<MeshNode*>(node0);
	impact.nodes[1] = const_cast<MeshNode*>(node1);
	impact.nodes[2] = const_cast<MeshNode*>(node2);
	impact.nodes[3] = const_cast<MeshNode*>(node3);

	Tensor x0{ node0->x_prev };
	Tensor v0{ node0->x - x0 };
	Tensor x1{ node1->x_prev - x0 };
	Tensor x2{ node2->x_prev - x0 };
	Tensor x3{ node3->x_prev - x0 };
	Tensor v1{ node1->x - node1->x_prev - v0 };
	Tensor v2{ node2->x - node2->x_prev - v0 };
	Tensor v3{ node3->x - node3->x_prev - v0 };

	Tensor a0{ torch::mm(x1.transpose(0, 1), torch::cross(x2, x3)) };
	Tensor a1{ torch::mm(v1.transpose(0, 1), torch::cross(x2, x3)) + 
		torch::mm(x1.transpose(0, 1), torch::cross(v2, x3)) + 
		torch::mm(x1.transpose(0, 1), torch::cross(x2, v3)) };
	Tensor a2{ torch::mm(x1.transpose(0, 1), torch::cross(v2, v3)) +
		torch::mm(v1.transpose(0, 1), torch::cross(x2, v3)) +
		torch::mm(v1.transpose(0, 1), torch::cross(v2, x3)) };
	Tensor a3{ torch::mm(v1.transpose(0, 1), torch::cross(v2, v3)) };

	Tensor t = CubicSolver::apply(a3, a2, a1, a0) ;

	int num_solutions = t.size(0);

	for (int i = 0; i < num_solutions; ++i) {
		if (torch::isnan(t[i]).item<bool>())
			continue;
		if (torch::isinf(t[i]).item<bool>())
			continue;
		if (t[i].item<double>() < 0 || t[i].item<double>() > 1) // Why greater than 1 ** Need to change**
			continue;
		impact.t = t[i];
		Tensor bx0{ x0 + t[i] * v0 };
		Tensor bx1{ x1 + t[i] * v1 };
		Tensor bx2{ x2 + t[i] * v2 };
		Tensor bx3{ x3 + t[i] * v3 };

		bool inside{ false }, over{ false };

		if (type == ColliType::VF) {
			sub_signed_vf_distance(bx1, bx2, bx3, impact.n, impact.ws, 1e-12, over);
			inside = torch::min(-impact.ws[1], torch::min(-impact.ws[2], -impact.ws[3])).item<double>() >= -1e-12;
		}
		else {
			sub_signed_ee_distance(bx1, bx2, bx3, bx2 - bx1, bx3 - bx1, bx3 - bx2, impact.n, impact.ws, 1e-12, over);
			inside = torch::min(torch::min(impact.ws[0], impact.ws[1]), torch::min(-impact.ws[2], -impact.ws[3])).item<double>() >= -1e-12;
		}

		if (over || !inside)
			continue;

		if (torch::dot(impact.n.squeeze(), (impact.ws[1] * v1 + impact.ws[2] * v2 + impact.ws[3] * v3).squeeze()).item<double>() > 0.0) {
			impact.n = -1 * impact.n;
		}

		return true;
	}
	return false;
}

void Collision::add_impacts(std::vector<ImpactZone*>& zones) {
	for (int z = 0; z < zones.size(); ++z) 
		zones[z]->active = false;
	for (int i = 0; i < impacts.size(); i++) {
		const Impact& impact = impacts[i];
		MeshNode* node = impact.nodes[is_free(impact.nodes[0]) ? 0:3];
		ImpactZone* zone{ find_or_create_zone(node, zones) };
		for (int n = 0; n < 4; n++)
			if (is_free(impact.nodes[n]))
				merge_zone(zone, find_or_create_zone(impact.nodes[n], zones), zones);
		
		zone->impacts.push_back(impact);
		zone->active = true;
	}
}

ImpactZone* Collision::find_or_create_zone(const MeshNode* node, std::vector<ImpactZone*>& zones){
	for (int z = 0; z < zones.size(); ++z) {
		if (is_in(node, zones[z]->nodes))
			return zones[z];
	}
	ImpactZone* zone = new ImpactZone;
	zone->nodes.push_back(const_cast<MeshNode*>(node));
	zones.push_back(zone);
	return zone;
}

void Collision::merge_zone(ImpactZone* zone0, ImpactZone* zone1, std::vector<ImpactZone*>& zones) {
	if (zone0 == zone1)
		return;
	zone0->nodes.insert(zone0->nodes.end(), zone1->nodes.begin(), zone1->nodes.end());
	zone0->impacts.insert(zone0->impacts.end(), zone1->impacts.begin(), zone1->impacts.end());
	for (int z = 0; z < zones.size(); ++z) {
		if (zone1 == zones[z])
			remove(z, zones);
	}
}

void precompute_derivative(real_2d_array& a, real_2d_array& q, real_2d_array& r0, std::vector<double>& lambda,
	real_1d_array& sm_1, std::vector<int>& legals, double** grads, ImpactZone* zone,
	NormalOpt& slx) {
	a.setlength(slx.nvar, legals.size());
	sm_1.setlength(slx.nvar);
	for (int i = 0; i < slx.nvar; ++i)
		sm_1[i] = 1.0 / sqrt(slx.inv_m * (zone->nodes[i / 3]->m)).item<double>();
	for (int k = 0; k < legals.size(); ++k)
		for (int i = 0; i < slx.nvar; ++i)
			a[i][k] = grads[legals[k]][i] * sm_1[i]; //sqrt(m^-1)
	real_1d_array tau, r1lam1, lamp;
	tau.setlength(slx.nvar);
	rmatrixqr(a, slx.nvar, legals.size(), tau);
	real_2d_array qtmp, r, r1;
	int cols = legals.size();
	if (cols > slx.nvar)cols = slx.nvar;
	rmatrixqrunpackq(a, slx.nvar, legals.size(), tau, cols, qtmp);
	rmatrixqrunpackr(a, slx.nvar, legals.size(), r);
	// get rid of degenerate G
	int newdim = 0;
	for (; newdim < cols; ++newdim)
		if (abs(r[newdim][newdim]) < 1e-6)
			break;
	r0.setlength(newdim, newdim);
	r1.setlength(newdim, legals.size() - newdim);
	q.setlength(slx.nvar, newdim);
	for (int i = 0; i < slx.nvar; ++i)
		for (int j = 0; j < newdim; ++j)
			q[i][j] = qtmp[i][j];
	for (int i = 0; i < newdim; ++i) {
		for (int j = 0; j < newdim; ++j)
			r0[i][j] = r[i][j];
		for (int j = newdim; j < legals.size(); ++j)
			r1[i][j - newdim] = r[i][j];
	}
	r1lam1.setlength(newdim);
	for (int i = 0; i < newdim; ++i) {
		r1lam1[i] = 0;
		for (int j = newdim; j < legals.size(); ++j)
			r1lam1[i] += r1[i][j - newdim] * lambda[legals[j]];
	}
	ae_int_t info;
	alglib::densesolverreport rep;
	rmatrixsolve(r0, (ae_int_t)newdim, r1lam1, info, rep, lamp);
	for (int j = 0; j < newdim; ++j)
		lambda[legals[j]] += lamp[j];
	for (int j = newdim; j < legals.size(); ++j)
		lambda[legals[j]] = 0;
}

std::vector<Tensor> apply_inelastic_projection_forward(Tensor xold, Tensor ws, Tensor ns, ImpactZone* zone, const Tensor& thickness) {
	NormalOpt slx{ zone };
	
	double *x = new double[slx.nvar];
	slx.initialize(x);
	int sign;
	std::vector<double> lambda = augmented_lagrangian_method(slx, thickness);

	std::vector<int> legals;
	double** grads = new double*[slx.ncon];
	double tmp;
	for (int i = 0; i < slx.ncon; ++i) {
		tmp = slx.constraint(&slx.tmp[0], i, sign, thickness);
		grads[i] = nullptr;
		if (sign == 1 && tmp > 1e-6) continue;//sign==1:tmp>=0
		if (sign == -1 && tmp < -1e-6) continue;
		grads[i] = new double[slx.nvar];
		for (int j = 0; j < slx.nvar; ++j)
			grads[i][j] = 0;
		slx.con_grad(&slx.tmp[0], i, 1, grads[i]);
		legals.push_back(i);
	}
	real_2d_array a, q, r;
	real_1d_array sm_1;//sqrt(m^-1)
	precompute_derivative(a, q, r, lambda, sm_1, legals, grads, zone, slx);
	Tensor q_tn = arr2ten(q), r_tn = arr2ten(r);
	Tensor lam_tn = ptr2ten(&lambda[0], lambda.size());
	Tensor sm1_tn = ptr2ten(sm_1.getcontent(), sm_1.length());
	Tensor legals_tn = ptr2ten(&legals[0], legals.size());
	Tensor ans = ptr2ten(&slx.tmp[0], slx.nvar);
	for (int i = 0; i < slx.ncon; ++i) {
		delete[] grads[i];
	}
	delete x;
	return { ans.reshape({-1, 3}), q_tn, r_tn, lam_tn, sm1_tn, legals_tn };

}

void apply_inelastic_projection(ImpactZone* zone, const Tensor& thickness) {

	Tensor inp_xold, inp_w, inp_n;

	std::vector<Tensor> xolds(zone->nodes.size());
	std::vector<Tensor> ws(zone->impacts.size() * 4);
	std::vector<Tensor> ns(zone->impacts.size());

	for (int i = 0; i < zone->nodes.size(); ++i) {
		xolds[i] = zone->nodes[i]->x.squeeze();
	}

	for (int j = 0; j < zone->impacts.size(); ++j) {
		ns[j] = zone->impacts[j].n.squeeze();
		for (int k = 0; k < 4; ++k) {
			ws[j * 4 + k] = zone->impacts[j].ws[k].squeeze();
		}
	}

	inp_xold = torch::stack(xolds);
	inp_w = torch::stack(ws);
	inp_n = torch::stack(ns);

	double* dw = inp_w.data_ptr<double>();
	double* dn = inp_n.data_ptr<double>();
	zone->w = std::vector<double>(dw, dw + zone->impacts.size() * 4);
	zone->n = std::vector<double>(dn, dn + zone->impacts.size() * 3);

	//Tensor out_x = apply_inelastic_projection_forward(inp_xold, inp_w, inp_n, zone, thickness)[0];

	Tensor out_x = InelasticProjection::apply(inp_xold, inp_w, inp_n, zone, thickness); // Can Backward

	for (int i = 0; i < zone->nodes.size(); ++i) {
		zone->nodes[i]->x = out_x[i].view({3,1});
	}
}

std::vector<Tensor> compute_derivative(real_1d_array& ans, ImpactZone* zone,
	real_2d_array& q, real_2d_array& r, real_1d_array& sm_1, std::vector<int>& legals,
	real_1d_array& dldx, std::vector<double>& lambda, bool verbose = false) {
	real_1d_array qtx, dz, dlam0, dlam, ana, dldw0, dldn0;
	int nvar = zone->nodes.size() * 3;
	int ncon = zone->impacts.size();
	qtx.setlength(q.cols());
	ana.setlength(nvar);
	dldn0.setlength(ncon * 3);
	dldw0.setlength(ncon * 4);
	dz.setlength(nvar);
	dlam0.setlength(q.cols());
	dlam.setlength(ncon);
	for (int i = 0; i < nvar; ++i)
		ana[i] = dz[i] = 0;
	for (int i = 0; i < ncon * 3; ++i) dldn0[i] = 0;
	for (int i = 0; i < ncon * 4; ++i) dldw0[i] = 0;
	// qtx = qt * sqrt(m^-1) dldx
	for (int i = 0; i < q.cols(); ++i) {
		qtx[i] = 0;
		for (int j = 0; j < nvar; ++j)
			qtx[i] += q[j][i] * dldx[j] * sm_1[j];
	}
	// dz = sqrt(m^-1) (sqrt(m^-1) dldx - q * qtx)
	for (int i = 0; i < nvar; ++i) {
		dz[i] = dldx[i] * sm_1[i];
		for (int j = 0; j < q.cols(); ++j)
			dz[i] -= q[i][j] * qtx[j];
		dz[i] *= sm_1[i];
	}
	// dlam = R^-1 * qtx
	ae_int_t info;
	alglib::densesolverreport rep;
	std::cout << "orisize=" << nvar << " " << ncon << " " << nvar + ncon;
	std::cout << "  size=" << q.cols() << std::endl;	
	rmatrixsolve(r, (ae_int_t)q.cols(), qtx, info, rep, dlam0);
	// cout<<endl;
	for (int j = 0; j < ncon; ++j)
		dlam[j] = 0;	
	for (int k = 0; k < q.cols(); ++k)
		dlam[legals[k]] = dlam0[k];
	//part1: dldq * dqdxt = M dz
	for (int i = 0; i < nvar; ++i)
		ana[i] += dz[i] / sm_1[i] / sm_1[i];
	//part2: dldg * dgdw * dwdxt
	for (int j = 0; j < ncon; ++j) {
		Impact& imp = zone->impacts[j];
		double* dldn = dldn0.getcontent() + j * 3;
		for (int n = 0; n < 4; n++) {
			int i = find(imp.nodes[n], zone->nodes);
			double& dldw = dldw0[j * 4 + n];
			if (i != -1) {
				for (int k = 0; k < 3; ++k) {
					//g=-w*n*x
					dldw += (dlam[j] * ans[i * 3 + k] + lambda[j] * dz[i * 3 + k]) * imp.n[k].item<double>();
					//part3: dldg * dgdn * dndxt
					dldn[k] += imp.ws[n].item<double>() * (dlam[j] * ans[i * 3 + k] + lambda[j] * dz[i * 3 + k]);
				}
			}
			else {
				//part4: dldh * (dhdw + dhdn)
				for (int k = 0; k < 3; ++k) {
					dldw += (dlam[j] * imp.n[k] * imp.nodes[n]->x[k]).item<double>();
					dldn[k] += (dlam[j] * imp.ws[n] * imp.nodes[n]->x[k]).item<double>();
				}
			}
		}
	}
	Tensor grad_xold = torch::from_blob(ana.getcontent(), { nvar / 3, 3 }, opts).clone();
	Tensor grad_w = torch::from_blob(dldw0.getcontent(), { ncon * 4 }, opts).clone();
	Tensor grad_n = torch::from_blob(dldn0.getcontent(), { ncon, 3 }, opts).clone();
	delete zone;
	return { grad_xold, grad_w, grad_n };
}

std::vector<Tensor> apply_inelastic_projection_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn, Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, ImpactZone* zone) {
	real_2d_array q = ten2arr(q_tn), r = ten2arr(r_tn);
	real_1d_array sm_1 = ten1arr(sm1_tn), ans = ten1arr(ans_tn.reshape({ -1 })), dldx = ten1arr(dldx_tn.reshape({ -1 }));
	std::vector<double> lambda = ten2vec<double>(lam_tn);
	std::vector<int> legals = ten2vec<int>(legals_tn);	
	return compute_derivative(ans, zone, q, r, sm_1, legals, dldx, lambda);
}

void NormalOpt::initialize(double* x) const {
	for (int n = 0; n < zone->nodes.size(); ++n)
		set_subvec(x, n, zone->nodes[n]->x);
}

void NormalOpt::precompute(const double* x) const {
	for (int n = 0; n < zone->nodes.size(); n++)
		zone->nodes[n]->x = get_subvec(x, n);
}

double NormalOpt::objective(const double* x) const {
	double e = 0;
	for (int n = 0; n < zone->nodes.size(); n++) {
		const MeshNode* node = zone->nodes[n];
		Tensor dx = node->x.squeeze() - node->x_old.squeeze();
		e = e + (inv_m * node->m * torch::dot(dx, dx) / 2).item<double>();
	}
	return e;
}

void NormalOpt::obj_grad(const double* x, double* grad) const {
	for (int n = 0; n < zone->nodes.size(); n++) {
		const MeshNode* node = zone->nodes[n];
		Tensor dx = node->x.squeeze() - node->x_old.squeeze();
		set_subvec(grad, n, inv_m * node->m * dx);
	}
}

double NormalOpt::constraint(const double* x, int j, int& sign, const Tensor& thickness) const {
	sign = -1;
	double c = thickness.item<double>();
	const Impact& impact = zone->impacts[j];
	for (int n = 0; n < 4; n++)
	{
		double* dx = impact.nodes[n]->x.data_ptr<double>();
		for (int k = 0; k < 3; ++k) {
			c -= zone->w[j * 4 + n] * zone->n[j * 3 + k] * dx[k];
		}
	}
	return c;
}

void NormalOpt::con_grad(const double* x, int j, double factor, double* grad) const {
	const Impact& impact = zone->impacts[j];
	for (int n = 0; n < 4; n++) {
		int i = find(impact.nodes[n], zone->nodes);
		if (i != -1)
			for (int k = 0; k < 3; ++k)
				grad[i * 3 + k] -= factor * zone->w[j * 4 + n] * zone->n[j * 3 + k];
	}
}

void NormalOpt::finalize(const double* x) {
	precompute(x);
	for (int i = 0; i < nvar; ++i)
		tmp[i] = x[i];
}