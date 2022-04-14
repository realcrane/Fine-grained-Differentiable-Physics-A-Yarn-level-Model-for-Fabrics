#include "Cloth.h"

//Cloth::Cloth(int _width, int _length,
//	double _hight, double _L, double _R,
//	Tensor _G, Tensor _rho, Tensor _Y,
//	Tensor _B, Tensor _kc, Tensor _kf,
//	Tensor _df, Tensor _mu, Tensor _S,
//	Tensor h_stiff, Wind _wind, WovenPattern _woven_pattern, InitPose _init_pose, bool _is_para) {
//
//	// Homo Constructor
//	width = _width;
//	length = _length;
//	hight = _hight;
//	L = _L;
//	R = _R;
//	G = _G;
//	rho = _rho;
//	Y = _Y;
//	B = _B;
//	ks = Y * PI * pow(R, 2);
//	kb = B * PI * pow(R, 2);
//	kc = _kc;
//	kf = _kf;
//	df = _df;
//	mu = _mu;
//	S = _S;
//
//	jamming_thrd = 2.0 * std::asin(R / L);
//
//	handle_stiffness = h_stiff;
//	wind = _wind;
//
//	num_nodes = width * length;
//	num_cros_nodes = (width - 2) * (length - 2);
//	num_LLD = num_nodes * num_nodes;
//	num_ELD = num_nodes * num_cros_nodes * 2;
//	num_EED = num_cros_nodes * num_cros_nodes * 2 * 2;
//
//	Zeros_L = std::vector<Tensor>(num_nodes, ZERO31);
//	Zeros_E = std::vector<Tensor>(num_cros_nodes, ZERO21);
//
//	Zeros_LL = std::vector<Tensor>(num_LLD, ZERO33);
//	Zeros_EL = std::vector<Tensor>(num_ELD, ZERO13);
//	Zeros_LE = std::vector<Tensor>(num_ELD, ZERO31);
//	Zeros_EE = std::vector<Tensor>(num_EED, ZERO11);
//
//	clothMesh.isCloth = true;
//
//	woven_pattern = _woven_pattern;
//	is_para = _is_para;
//
//	if (is_para)
//		omp_set_num_threads(NThrd);	// Set OpenMP Thread Number to NThrd: Const defined in Cloth.h
//
//	build(_init_pose);
//
//	//if (is_para) {
//	//	std::cout << "Parallel Simulation" << std::endl;
//	//	InitMV_Para();
//	//	FillMV_Para();
//	//}
//	//else {
//	//	std::cout << "Sequence Simulation" << std::endl;
//	//	InitMV();
//	//	FillMV();		
//	//}
//
//	InitClothUnits();
//}


Cloth::Cloth(int _width, int _length, double _hight, double _L,
	double _R, std::vector<Yarn> _yarns, Tensor _kc, Tensor _kf, Tensor _df, Tensor _mu, Tensor _S,
	Tensor h_stiff, WovenPattern _woven_pattern, InitPose _init_pose) {

	width = _width;
	length = _length; 
	hight = _hight; 
	L = _L;
	R = _R;

	yarns = _yarns;

	if (yarns.size() == 1)
		std::cout << "Pure Woven" << std::endl;
	else if (yarns.size() > 1)
		std::cout << "Blend Woven" << std::endl;
		
	kc = _kc; 
	kf = _kf; 
	df = _df;
	mu = _mu;
	S = _S;
			
	handle_stiffness = h_stiff;
	woven_pattern = _woven_pattern;

	jamming_thrd = 2.0 * std::asin(R / L);

	num_nodes = width * length;
	num_cros_nodes = (width - 2) * (length - 2);
	num_LLD = num_nodes * num_nodes;
	num_ELD = num_nodes * num_cros_nodes * 2;
	num_EED = num_cros_nodes * num_cros_nodes * 2 * 2;

	Zeros_L = std::vector<Tensor>(num_nodes, ZERO31);
	Zeros_E = std::vector<Tensor>(num_cros_nodes, ZERO21);

	Zeros_LL = std::vector<Tensor>(num_LLD, ZERO33);
	Zeros_EL = std::vector<Tensor>(num_ELD, ZERO13);
	Zeros_LE = std::vector<Tensor>(num_ELD, ZERO31);
	Zeros_EE = std::vector<Tensor>(num_EED, ZERO11);

	clothMesh.isCloth = true;

	build(_init_pose);

	InitClothUnits();
}

void Cloth::build(InitPose init_pose) {
	for (int j = 0; j < length; ++j)
		for (int i = 0; i < width; ++i)
		{	
			std::string idx = std::to_string(i) + "," + std::to_string(j);
			Tensor LPos = torch::tensor({ i * L, j * L, hight }, opts).view({ 3,1 });
			if (init_pose == InitPose::Upright)
				LPos = torch::tensor({ i * L, 0.0, hight - j * L }, opts).view({ 3,1 });
			Tensor EPos = torch::tensor({ i * L, j * L }, opts).view({ 2,1 });
			Tensor LVel = ZERO31;
			Tensor EVel = ZERO21;
			bool flip_norm{ false };
			// Yarn nodes
			if (i == 0 || i == width - 1 || j == 0 || j == length - 1)
				nodes.insert(std::pair<std::string, Node>(idx, Node(i, j, LPos, EPos, LVel, EVel, LPos.clone().detach(), EPos, true, flip_norm)));
			else {

				switch (woven_pattern) {
				case(WovenPattern::Plain): {
					if ((i + j) % 2 != 0)
						flip_norm = true;
				}break;
				case(WovenPattern::Twill): {
					int reminder_v = j % 4;
					int reminder_u = (i + reminder_v) % 4;
					if (reminder_u < 2)
						flip_norm = true;
				}break;
				case(WovenPattern::Satin): {
					int reminder_v = (j - 1) % 4;
					int quotient_unit_v = reminder_v / 2;
					int reminder_unit_v = reminder_v % 2;
					int index_unit_v = quotient_unit_v * 3 + reminder_unit_v * 2;
					int reminder_u = (i - 1 + index_unit_v) % 4;
					if (reminder_u == 0)
						flip_norm = true;
				}break;
				default: {
					std::cerr << "Unknow Woven Pattern" << std::endl;
				}
				}

				nodes.insert(std::pair<std::string, Node>(idx, Node(i, j, LPos, EPos, LVel, EVel, LPos.clone().detach(), EPos, false, flip_norm)));
			}
			// Mesh verts and nodes
			clothMesh.verts.push_back(MeshVert(EPos));
			clothMesh.nodes.push_back(MeshNode(LPos, LVel));
		}
	// Connect vert and node
	for (int n = 0; n < clothMesh.nodes.size(); ++n) {
		clothMesh.verts.at(n).index = n;
		clothMesh.nodes.at(n).index = n;
		clothMesh.nodes.at(n).m = torch::zeros({ 1 }, opts);
		connect(&clothMesh.verts.at(n), &clothMesh.nodes.at(n));
	}
	warp_nodes = nodes;
	weft_nodes = nodes;	
}

void Cloth::update(const Tensor& new_vel, const Tensor& F_E_is_Slide, const Tensor& h)
{
	for (int i = 0; i < width; ++i)
		for (int j = 0; j < length; ++j) {
			// Update Yarn Cloth
			std::string idx{ std::to_string(i) + "," + std::to_string(j) };
			int idx_L_begin{ 3 * (i + j * width) };
			int idx_mesh{ i + j * width };
			// Update Mesh Node Previous Position State
			clothMesh.nodes.at(idx_mesh).x_prev = nodes[idx].LPos;
			// Update Yarn Node Current Position State
			nodes[idx].LVel = new_vel.slice(0, idx_L_begin, idx_L_begin + 3, 1);
			nodes[idx].LPos = nodes[idx].LPos + nodes[idx].LVel * h;
			// Update warp and weft nodes position (With Crimp)
			warp_nodes[idx].LPos = nodes[idx].LPos;
			weft_nodes[idx].LPos = nodes[idx].LPos;
			// Update Mesh Cloth
			clothMesh.nodes.at(idx_mesh).m = torch::zeros({ 1 }, opts);
			clothMesh.nodes.at(idx_mesh).v = nodes[idx].LVel;
			clothMesh.nodes.at(idx_mesh).x = nodes[idx].LPos;
			// Update non-fixed yarn cross node
			if (!nodes[idx].is_fixed)
				nodes[idx].LPosFix = nodes[idx].LPos.clone().detach(); // ** Potential error while Learning
			// Updata Euler Pos and Vel
			if (!(nodes[idx].is_edge)) {
				// Update Yarn Cloth
				int idx_E{ (i - 1 + (j - 1) * (width - 2)) };
				int idx_E_begin{ num_nodes * 3 + idx_E * 2 };
				nodes[idx].EVel = new_vel.slice(0, idx_E_begin, idx_E_begin + 2, 1);;
				nodes[idx].EPos = nodes[idx].EPos + nodes[idx].EVel * h;
				nodes[idx].EPosBar = nodes[idx].EPosBar + F_E_is_Slide[idx_E] * nodes[idx].EVel * h; // Updata E Anchor If slide
				// Updata Mesh Cloth
				clothMesh.verts.at(idx_mesh).uv = nodes[idx].EPos;
			}
		}

	UpdateClothUnits();
}

void Cloth::update() {
	// Update cloth from clothMesh
	for (int i = 0; i < width; ++i)
		for (int j = 0; j < length; ++j) {
			std::string idx{ std::to_string(i) + "," + std::to_string(j) };
			int idx_mesh{ i + j * width };
			nodes[idx].LPos = clothMesh.nodes[idx_mesh].x;
			nodes[idx].LVel = clothMesh.nodes[idx_mesh].v;
			warp_nodes[idx].LPos = nodes[idx].LPos;
			weft_nodes[idx].LPos = nodes[idx].LPos;
			if (!nodes[idx].is_fixed)
				nodes[idx].LPosFix = nodes[idx].LPos.clone().detach();
		}

	UpdateClothUnits();
}

void Cloth::InitClothUnits()
{
	// warp, weft nodes, and shear segments
	for (int i = 1; i < width - 1; ++i)
		for (int j = 1; j < length - 1; ++j) {
			std::string idx = std::to_string(i) + "," + std::to_string(j);
			std::string idx_left = std::to_string(i - 1) + "," + std::to_string(j);
			std::string idx_right = std::to_string(i + 1) + "," + std::to_string(j);
			std::string idx_top = std::to_string(i) + "," + std::to_string(j - 1);
			std::string idx_bot = std::to_string(i) + "," + std::to_string(j + 1);

			shear_segs.push_back(Shear_Seg(&nodes[idx], &nodes[idx_left], &nodes[idx_bot]));
			shear_segs.push_back(Shear_Seg(&nodes[idx], &nodes[idx_left], &nodes[idx_top]));
			shear_segs.push_back(Shear_Seg(&nodes[idx], &nodes[idx_right], &nodes[idx_bot]));
			shear_segs.push_back(Shear_Seg(&nodes[idx], &nodes[idx_right], &nodes[idx_top]));

			Tensor norm = NodeNorm(nodes[idx], nodes[idx_left], nodes[idx_right], nodes[idx_top], nodes[idx_bot]);
			nodes[idx].n = norm;

			warp_nodes[idx].LPos = warp_nodes[idx].LPos - R * norm;
			weft_nodes[idx].LPos = weft_nodes[idx].LPos + R * norm;
		}
	// edge : warp
	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width - 1; ++j) {
			// Create and push back Yarn Edge : Along Warp
			std::string idx_n0 = std::to_string(j) + "," + std::to_string(i);
			std::string idx_n1 = std::to_string(j + 1) + "," + std::to_string(i);
			if (yarns.size() == 1) {	// Homogeneous
				edges.push_back(Edge(&nodes[idx_n0], &nodes[idx_n1], yarns[0], "warp"));
			}
			else {	// Heterogeneous
				if (i % 2 == 0)
					edges.push_back(Edge(&nodes[idx_n0], &nodes[idx_n1], yarns[0], "warp"));
				else
					edges.push_back(Edge(&nodes[idx_n0], &nodes[idx_n1], yarns[1], "warp"));
			}
			// Create and push back Mesh Edge
			int idx_mesh_n0{ j + i * width };
			int idx_mesh_n1{ (j + 1) + i * width };
			clothMesh.add(MeshEdge(&clothMesh.nodes.at(idx_mesh_n0), &clothMesh.nodes.at(idx_mesh_n1)));
		}
	// edge : weft
	for (int i = 0; i < width; ++i)
		for (int j = 0; j < length - 1; ++j) {
			// Create and push back Yarn Edge : Along Weft
			std::string idx_n0 = std::to_string(i) + "," + std::to_string(j);
			std::string idx_n1 = std::to_string(i) + "," + std::to_string(j + 1);
			if (yarns.size() == 1) {	// Homogeneous
				edges.push_back(Edge(&nodes[idx_n0], &nodes[idx_n1], yarns[0], "weft"));
			}
			else {	// Heterogeneous
				if (i % 2 == 0)
					edges.push_back(Edge(&nodes[idx_n0], &nodes[idx_n1], yarns[0], "weft"));
				else
					edges.push_back(Edge(&nodes[idx_n0], &nodes[idx_n1], yarns[1], "weft"));
			}
			// Create and push back Mesh Edge
			int idx_mesh_n0{ i + j * width };
			int idx_mesh_n1{ i + (j + 1) * width };
			clothMesh.add(MeshEdge(&clothMesh.nodes.at(idx_mesh_n0), &clothMesh.nodes.at(idx_mesh_n1)));
		}
	// bend edge : warp
	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width - 2; ++j) {
			std::string idx_n2 = std::to_string(j) + "," + std::to_string(i);
			std::string idx_n0 = std::to_string(j + 1) + "," + std::to_string(i);
			std::string idx_n1 = std::to_string(j + 2) + "," + std::to_string(i);
			if (yarns.size() == 1) {
				bend_segs.push_back(Bend_Seg(&nodes[idx_n2], &nodes[idx_n0], &nodes[idx_n1], yarns[0], "warp"));
				crimp_bend_segs.push_back(Bend_Seg(&warp_nodes[idx_n2], &warp_nodes[idx_n0], &warp_nodes[idx_n1], yarns[0], "warp"));
			}
			else {
				if (i % 2 == 0) {
					bend_segs.push_back(Bend_Seg(&nodes[idx_n2], &nodes[idx_n0], &nodes[idx_n1], yarns[0], "warp"));
					crimp_bend_segs.push_back(Bend_Seg(&warp_nodes[idx_n2], &warp_nodes[idx_n0], &warp_nodes[idx_n1], yarns[0], "warp"));
				}
				else {
					bend_segs.push_back(Bend_Seg(&nodes[idx_n2], &nodes[idx_n0], &nodes[idx_n1], yarns[1], "warp"));
					crimp_bend_segs.push_back(Bend_Seg(&warp_nodes[idx_n2], &warp_nodes[idx_n0], &warp_nodes[idx_n1], yarns[1], "warp"));
				}
			}
		}
	// bend edge : weft
	for (int i = 0; i < width; ++i)
		for (int j = 0; j < length - 2; ++j) {
			std::string idx_n2 = std::to_string(i) + "," + std::to_string(j);
			std::string idx_n0 = std::to_string(i) + "," + std::to_string(j + 1);
			std::string idx_n1 = std::to_string(i) + "," + std::to_string(j + 2);
			if (yarns.size() == 1) {
				bend_segs.push_back(Bend_Seg(&nodes[idx_n2], &nodes[idx_n0], &nodes[idx_n1], yarns[0], "weft"));
				crimp_bend_segs.push_back(Bend_Seg(&weft_nodes[idx_n2], &weft_nodes[idx_n0], &weft_nodes[idx_n1], yarns[0], "weft"));
			}
			else {
				if (i % 2 == 0) {
					bend_segs.push_back(Bend_Seg(&nodes[idx_n2], &nodes[idx_n0], &nodes[idx_n1], yarns[0], "weft"));
					crimp_bend_segs.push_back(Bend_Seg(&weft_nodes[idx_n2], &weft_nodes[idx_n0], &weft_nodes[idx_n1], yarns[0], "weft"));
				}
				else {
					bend_segs.push_back(Bend_Seg(&nodes[idx_n2], &nodes[idx_n0], &nodes[idx_n1], yarns[1], "weft"));
					crimp_bend_segs.push_back(Bend_Seg(&weft_nodes[idx_n2], &weft_nodes[idx_n0], &weft_nodes[idx_n1], yarns[1], "weft"));
				}
			}
		}
	// face
	for (int i = 0; i < width - 1; ++i)
		for (int j = 0; j < length - 1; ++j) {
			// Yarn Face
			std::string idx_top_left = std::to_string(i) + "," + std::to_string(j);
			std::string idx_top_right = std::to_string(i + 1) + "," + std::to_string(j);
			std::string idx_bot_left = std::to_string(i) + "," + std::to_string(j + 1);
			std::string idx_bot_right = std::to_string(i + 1) + "," + std::to_string(j + 1);
			if ((i + j) % 2 == 0) {
				faces.push_back(Face(&nodes[idx_top_left], &nodes[idx_top_right], &nodes[idx_bot_right]));
				faces.push_back(Face(&nodes[idx_top_left], &nodes[idx_bot_right], &nodes[idx_bot_left]));
			}
			else {
				faces.push_back(Face(&nodes[idx_top_left], &nodes[idx_top_right], &nodes[idx_bot_left]));
				faces.push_back(Face(&nodes[idx_top_right], &nodes[idx_bot_right], &nodes[idx_bot_left]));
			}

			// Mesh Face
			int idx_mesh_top_left{ i + j * width };
			int idx_mesh_top_right{ i + 1 + j * width };
			int idx_mesh_bot_left{ i + (j + 1) * width };
			int idx_mesh_bot_right{ (i + 1) + (j + 1) * width };
			clothMesh.add(MeshEdge(&clothMesh.nodes.at(idx_mesh_top_left), &clothMesh.nodes.at(idx_mesh_bot_right)));
			clothMesh.add(MeshFace(&clothMesh.verts.at(idx_mesh_top_left), &clothMesh.verts.at(idx_mesh_top_right), &clothMesh.verts.at(idx_mesh_bot_right)));
			clothMesh.add(MeshFace(&clothMesh.verts.at(idx_mesh_top_left), &clothMesh.verts.at(idx_mesh_bot_right), &clothMesh.verts.at(idx_mesh_bot_left)));
		}

	ConnectNodeFace();
	clothMesh.add_adjecent();

	for (int i = 0; i < width; ++i)
		for (int j = 0; j < length; ++j) {
			MeshMass(i, j);
		}
}

void Cloth::UpdateClothUnits()
{
#pragma omp parallel
	{
#pragma omp for collapse(2)
		for (int i = 1; i < width - 1; ++i)
			for (int j = 1; j < length - 1; ++j)
			{
				std::string idx = std::to_string(i) + "," + std::to_string(j);
				std::string idx_left = std::to_string(i - 1) + "," + std::to_string(j);
				std::string idx_right = std::to_string(i + 1) + "," + std::to_string(j);
				std::string idx_top = std::to_string(i) + "," + std::to_string(j - 1);
				std::string idx_bot = std::to_string(i) + "," + std::to_string(j + 1);

				Tensor norm = NodeNorm(nodes[idx], nodes[idx_left], nodes[idx_right], nodes[idx_top], nodes[idx_bot]);
				nodes[idx].n = norm;

				warp_nodes[idx].LPos = nodes[idx].LPos - R * norm;
				weft_nodes[idx].LPos = nodes[idx].LPos + R * norm;
			}

#pragma omp for nowait
		for (int e = 0; e < edges.size(); ++e)
			edges[e].update(); // Update Edges
#pragma omp for	nowait
		for (int b = 0; b < bend_segs.size(); ++b)
			bend_segs[b].update(); // Update Bend Segments
#pragma omp for nowait
		for (int b = 0; b < crimp_bend_segs.size(); ++b)
			crimp_bend_segs[b].update(); // Update Crimp Bend Segments
#pragma omp for nowait
		for (int s = 0; s < shear_segs.size(); ++s)
			shear_segs[s].update(); // Update Shear Segments
#pragma omp for	nowait		
		for (int f = 0; f < faces.size(); ++f)
			faces[f].update(); // Update Faces
	}
	for (int i = 0; i < width; ++i)
		for (int j = 0; j < length; ++j) {
			MeshMass(i, j);
		}
}

ClothState Cloth::get_ClothState() const {
	Tensor LPos{ torch::zeros({width * length, 3, 1}, opts) };
	Tensor LPosFix{ torch::zeros({width * length, 3, 1}, opts) };
	Tensor LVel{ torch::zeros({width * length, 3, 1}, opts) };

	Tensor EPos{ torch::zeros({width * length, 2, 1}, opts) };
	Tensor EPosBar{ torch::zeros({width * length, 2, 1}, opts) };
	Tensor EVel{ torch::zeros({width * length, 2, 1}, opts) };

	for (int v = 0; v < length; ++v)
		for (int u = 0; u < width; ++u) {
			std::string node_idx = std::to_string(u) + "," + std::to_string(v);
			int state_idx = (u + v * length);
			LPos[state_idx] = nodes.at(node_idx).LPos;
			LPosFix[state_idx] = nodes.at(node_idx).LPosFix;
			LVel[state_idx] = nodes.at(node_idx).LVel;
			EPos[state_idx] = nodes.at(node_idx).EPos;
			EPosBar[state_idx] = nodes.at(node_idx).EPosBar;
			EVel[state_idx] = nodes.at(node_idx).EVel;
		}
	return { LPos, LPosFix, LVel, EPos, EPosBar, EVel };
}

void Cloth::ConnectNodeFace() {

	auto iter = nodes.begin();

	while(iter != nodes.end())
	{
		for (int f = 0; f < faces.size(); ++f)
			if (faces[f].is_include(&(iter->second)))
				iter->second.adj_faces.insert(&faces[f]);			
		iter++;
	}
}

void Cloth::set_handles(const std::vector<std::pair<int, int>>& idxs) {
	// Set handle nodes
	for (const auto& idx : idxs) {
		std::string n_idx = std::to_string(idx.first) + "," + std::to_string(idx.second);
		int idx_mesh{ idx.first + idx.second * width };
		nodes[n_idx].is_fixed = true;
		clothMesh.nodes.at(idx_mesh).is_fixed = true;
	}
}

void Cloth::set_ClothStateMesh(const ClothState& state) {
	// Set cloth state from h5 file data.
	// Set cloth mesh information for collision processing
	for (int v = 0; v < length; ++v)
		for (int u = 0; u < width; ++u) {
			std::string node_idx = std::to_string(u) + "," + std::to_string(v);
			int state_idx = (u + v * length);
			nodes[node_idx].LPos = state.LPos[state_idx];
			warp_nodes[node_idx].LPos = state.LPos[state_idx];
			weft_nodes[node_idx].LPos = state.LPos[state_idx];
			nodes[node_idx].LPosFix = state.LPosFix[state_idx];
			nodes[node_idx].LVel = state.LVel[state_idx];
			nodes[node_idx].EPos = state.EPos[state_idx];
			nodes[node_idx].EPosBar = state.EPosBar[state_idx];
			nodes[node_idx].EVel = state.EVel[state_idx];

			clothMesh.verts[state_idx].uv = state.EPos[state_idx];
			clothMesh.nodes[state_idx].x = state.LPos[state_idx];
			clothMesh.nodes[state_idx].x_prev = state.LPos[state_idx];
			clothMesh.nodes[state_idx].v = state.LVel[state_idx];
		}

	UpdateClothUnits();
}

void Cloth::set_ClothState(const ClothState& state) {
	// Set cloth state from h5 file data.
	for (int v = 0; v < length; ++v)
		for (int u = 0; u < width; ++u) {
			std::string node_idx = std::to_string(u) + "," + std::to_string(v);
			int state_idx = (u + v * length);
			nodes[node_idx].LPos = state.LPos[state_idx];
			warp_nodes[node_idx].LPos = state.LPos[state_idx];
			weft_nodes[node_idx].LPos = state.LPos[state_idx];
			nodes[node_idx].LPosFix = state.LPosFix[state_idx];
			nodes[node_idx].LVel = state.LVel[state_idx];
			nodes[node_idx].EPos = state.EPos[state_idx];
			nodes[node_idx].EPosBar = state.EPosBar[state_idx];
			nodes[node_idx].EVel = state.EVel[state_idx];
		}

	UpdateClothUnits();
}

Tensor Cloth::NodeNorm(const Node& n_cen, const Node& n_left, const Node& n_right, const Node& n_top, const Node& n_bot) {

	Tensor A{ torch::cat({ n_cen.LPos, n_left.LPos, n_right.LPos, n_top.LPos, n_bot.LPos }, 1).transpose(1,0) };
	Tensor A_mean{ torch::mean(A, 0).view({ 1,3 }) };
	Tensor A_sub_mean{ A - A_mean };

	auto svd = torch::svd(A_sub_mean, false, true);

	Tensor normal_raw{ torch::slice(std::get<2>(svd), 1, 2) };
	Tensor normal{ normal_raw / torch::norm(normal_raw) };

	Tensor vec_warp{ n_right.LPos - n_left.LPos };
	Tensor vec_weft{ n_bot.LPos - n_top.LPos };
	Tensor vec_cross{ torch::transpose(at::cross(vec_warp, vec_weft), 1, 0) };
	Tensor normal_cos{ torch::mm(vec_cross, normal) / (torch::norm(vec_cross) * torch::norm(normal)) };

	if (normal_cos.item<double>() < 0)
		normal = -normal;

	if (n_cen.flip_norm)
		normal = -normal;

	return normal;
}

void Cloth::MeshMass(int u, int v) {
	int node_idx{ u + v * width };
	if (yarns.size() == 1) {
		clothMesh.nodes[node_idx].m = yarns[0].rho;
	}
	else if (yarns.size() == 2) {
		clothMesh.nodes[node_idx].m = 0.5 * (yarns[0].rho + yarns[1].rho);
	}
}
