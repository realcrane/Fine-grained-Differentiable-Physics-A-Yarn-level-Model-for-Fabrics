#include "Loader.h"

std::pair<H5ClothSize, H5Physical> cloth_property_loader(std::string filepath) {

	H5File file(filepath, H5F_ACC_RDONLY);

	DataSet dataset_size = file.openDataSet("/Properties/Size");

	CompType cloth_size(sizeof(H5ClothSize));

	cloth_size.insertMember("width", HOFFSET(H5ClothSize, width), PredType::NATIVE_INT);
	cloth_size.insertMember("length", HOFFSET(H5ClothSize, length), PredType::NATIVE_INT);

	H5ClothSize clothsize[1];
	dataset_size.read(clothsize, cloth_size);

	dataset_size.close();

	DataSet dataset_properties = file.openDataSet("/Properties/Physical properties");

	CompType cloth_physical(sizeof(H5Physical));

	cloth_physical.insertMember("Gravity", HOFFSET(H5Physical, gravity), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("rho", HOFFSET(H5Physical, rho), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("R", HOFFSET(H5Physical, R), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("L", HOFFSET(H5Physical, L), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("Y", HOFFSET(H5Physical, Y), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("B", HOFFSET(H5Physical, B), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("kc", HOFFSET(H5Physical, kc), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("kf", HOFFSET(H5Physical, kf), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("df", HOFFSET(H5Physical, df), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("mu", HOFFSET(H5Physical, mu), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("S", HOFFSET(H5Physical, S), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("handle_stiff", HOFFSET(H5Physical, handle_stiffness), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_velocity_x", HOFFSET(H5Physical, wind_velocity_x), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_velocity_y", HOFFSET(H5Physical, wind_velocity_y), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_velocity_z", HOFFSET(H5Physical, wind_velocity_z), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_density", HOFFSET(H5Physical, wind_density), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_drag", HOFFSET(H5Physical, wind_drag), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("woven_pattern", HOFFSET(H5Physical, woven_pattern), PredType::NATIVE_INT);

	H5Physical physicals[1];
	dataset_properties.read(physicals, cloth_physical);

	dataset_properties.close();
	file.close();

	return std::make_pair(clothsize[0], physicals[0]);
}

std::pair<H5ClothSize, H5PhysicalHeterYarn> cloth_property_loader_heter(std::string filepath) {

	H5File file(filepath, H5F_ACC_RDONLY);

	DataSet dataset_size = file.openDataSet("/Properties/Size");

	CompType cloth_size(sizeof(H5ClothSize));

	cloth_size.insertMember("width", HOFFSET(H5ClothSize, width), PredType::NATIVE_INT);
	cloth_size.insertMember("length", HOFFSET(H5ClothSize, length), PredType::NATIVE_INT);

	H5ClothSize clothsize[1];
	dataset_size.read(clothsize, cloth_size);

	dataset_size.close();

	DataSet dataset_properties = file.openDataSet("/Properties/Physical properties");

	CompType cloth_physical(sizeof(H5PhysicalHeterYarn));

	cloth_physical.insertMember("Gravity", HOFFSET(H5PhysicalHeterYarn, gravity), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("Yarn1_rho", HOFFSET(H5PhysicalHeterYarn, rho1), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("Yarn2_rho", HOFFSET(H5PhysicalHeterYarn, rho2), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("R", HOFFSET(H5PhysicalHeterYarn, R), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("L", HOFFSET(H5PhysicalHeterYarn, L), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("Yarn1_Y", HOFFSET(H5PhysicalHeterYarn, Y1), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("Yarn2_Y", HOFFSET(H5PhysicalHeterYarn, Y2), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("Yarn1_B", HOFFSET(H5PhysicalHeterYarn, B1), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("Yarn2_B", HOFFSET(H5PhysicalHeterYarn, B2), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("kc", HOFFSET(H5PhysicalHeterYarn, kc), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("kf", HOFFSET(H5PhysicalHeterYarn, kf), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("df", HOFFSET(H5PhysicalHeterYarn, df), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("mu", HOFFSET(H5PhysicalHeterYarn, mu), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("S", HOFFSET(H5PhysicalHeterYarn, S), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("handle_stiff", HOFFSET(H5PhysicalHeterYarn, handle_stiffness), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_velocity_x", HOFFSET(H5PhysicalHeterYarn, wind_velocity_x), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_velocity_y", HOFFSET(H5PhysicalHeterYarn, wind_velocity_y), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_velocity_z", HOFFSET(H5PhysicalHeterYarn, wind_velocity_z), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_density", HOFFSET(H5PhysicalHeterYarn, wind_density), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("wind_drag", HOFFSET(H5PhysicalHeterYarn, wind_drag), PredType::NATIVE_DOUBLE);
	cloth_physical.insertMember("woven_pattern", HOFFSET(H5PhysicalHeterYarn, woven_pattern), PredType::NATIVE_INT);

	H5PhysicalHeterYarn physicals[1];
	dataset_properties.read(physicals, cloth_physical);

	dataset_properties.close();
	file.close();

	return std::make_pair(clothsize[0], physicals[0]);
}

ClothState data_loader(std::string filepath, int start, int end, torch::TensorOptions opt) {

	int steps = end - start;

	H5File file(filepath, H5F_ACC_RDONLY);

	DataSet dataset_size = file.openDataSet("/Properties/Size");

	CompType cloth_size(sizeof(H5ClothSize));

	cloth_size.insertMember("width", HOFFSET(H5ClothSize, width), PredType::NATIVE_INT);
	cloth_size.insertMember("length", HOFFSET(H5ClothSize, length), PredType::NATIVE_INT);

	H5ClothSize clothsize[1];
	dataset_size.read(clothsize, cloth_size);

	int width{ clothsize[0].width };
	int length{ clothsize[0].length };

	dataset_size.close();

	double Ldata_out[1][3];
	double Edata_out[1][2];

	hsize_t dimsm[2];
	dimsm[0] = 1;
	dimsm[1] = 3;

	hsize_t offset[2];
	hsize_t count[2];

	hsize_t offset_out[2];
	hsize_t count_out[2];

	Tensor LPos = torch::zeros({ steps, width * length, 3, 1 }, opt);
	Tensor LVel = torch::zeros({ steps, width * length, 3, 1 }, opt);

	Tensor EPos = torch::zeros({ steps, width * length, 2, 1 }, opt);
	Tensor EVel = torch::zeros({ steps, width * length, 2, 1 }, opt);

	Tensor LPosFix = torch::zeros({ steps, width * length, 3, 1 }, opt);
	Tensor EPosBar = torch::zeros({ steps, width * length, 2, 1 }, opt);

	for (int step = 0; step < steps; ++step)
		for (int v = 0; v < length; ++v)
			for (int u = 0; u < width; ++u) {
				
				dimsm[1] = 3;
				for (int i = 0; i < 3; ++i)
					Ldata_out[0][i] = 0.0;

				std::string LPosDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/LPOS";
				std::string LVelDatasetName = "/Steps/Velocity/Node " + std::to_string(u) + " " + std::to_string(v) + "/LVEL";
				std::string EPosDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/EPOS";
				std::string EVelDatasetName = "/Steps/Velocity/Node " + std::to_string(u) + " " + std::to_string(v) + "/EVEL";
				std::string LPosFixDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/LPOSFix";
				std::string EPosBarDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/EPOSBar";

				int tensor_idx = u + v * width;
				// Read L Pos
				std::unique_ptr<DataSet> dataset = std::make_unique<DataSet>(file.openDataSet(LPosDatasetName));
				DataSpace dataspace = dataset->getSpace();

				offset[0] = start + step;
				offset[1] = 0;
				count[0] = 1;
				count[1] = 3;
				dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

				DataSpace Lmemespace(2, dimsm);

				offset_out[0] = 0;
				offset_out[1] = 0;
				count_out[0] = 1;
				count_out[1] = 3;
				Lmemespace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);

				dataset->read(Ldata_out, PredType::NATIVE_DOUBLE, Lmemespace, dataspace);

				LPos[step][tensor_idx] = torch::from_blob(Ldata_out[0], { 3,1 }, opt);

				// Read L Pos Fix
				for (int i = 0; i < 3; ++i)
					Ldata_out[0][i] = 0.0;

				dataset = std::make_unique<DataSet>(file.openDataSet(LPosFixDatasetName));
				dataspace = dataset->getSpace();

				dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
				Lmemespace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);
				dataset->read(Ldata_out, PredType::NATIVE_DOUBLE, Lmemespace, dataspace);

				LPosFix[step][tensor_idx] = torch::from_blob(Ldata_out[0], { 3,1 }, opt);

				// Read L Vel
				for (int i = 0; i < 3; ++i)
					Ldata_out[0][i] = 0.0;

				dataset = std::make_unique<DataSet>(file.openDataSet(LVelDatasetName));
				dataspace = dataset->getSpace();

				dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
				Lmemespace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);
				dataset->read(Ldata_out, PredType::NATIVE_DOUBLE, Lmemespace, dataspace);

				LVel[step][tensor_idx] = torch::from_blob(Ldata_out[0], { 3,1 }, opt);

				// Read E Pos
				dimsm[1] = 2;
				for (int i = 0; i < 2; ++i)
					Edata_out[0][i] = 0.0;

				dataset = std::make_unique<DataSet>(file.openDataSet(EPosDatasetName));
				dataspace = dataset->getSpace();

				offset[0] = start + step;
				offset[1] = 0;
				count[0] = 1;
				count[1] = 2;
				dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

				DataSpace Ememespace(2, dimsm);

				offset_out[0] = 0;
				offset_out[1] = 0;
				count_out[0] = 1;
				count_out[1] = 2;
				Ememespace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);

				dataset->read(Edata_out, PredType::NATIVE_DOUBLE, Ememespace, dataspace);

				EPos[step][tensor_idx] = torch::from_blob(Edata_out[0], { 2,1 }, opt);

				// Read E Vel
				for (int i = 0; i < 2; ++i)
					Edata_out[0][i] = 0.0;

				dataset = std::make_unique<DataSet>(file.openDataSet(EVelDatasetName));
				dataspace = dataset->getSpace();

				dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
				Ememespace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);

				dataset->read(Edata_out, PredType::NATIVE_DOUBLE, Ememespace, dataspace);

				EVel[step][tensor_idx] = torch::from_blob(Edata_out, { 2,1 }, opt);

				// Read E Position Bar
				for (int i = 0; i < 2; ++i)
					Edata_out[0][i] = 0.0;

				dataset = std::make_unique<DataSet>(file.openDataSet(EPosBarDatasetName));
				dataspace = dataset->getSpace();

				dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
				Ememespace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);

				dataset->read(Edata_out, PredType::NATIVE_DOUBLE, Ememespace, dataspace);

				EPosBar[step][tensor_idx] = torch::from_blob(Edata_out, { 2,1 }, opt);
			}

	file.close();

	return { LPos, LPosFix, LVel, EPos, EPosBar, EVel };
}

void load_obj(std::string filepath, Mesh& mesh) {
	std::ifstream f;
	f.open(filepath);

	std::string line{};
	int idx_node{ 0 }, idx_vert{ 0 }, idx_face{ 0 };

	std::vector<std::array<int, 3>> faces_verts;
	std::vector<std::array<int, 3>> faces_nodes;
	std::set<std::pair<int, int>> v_vt_pair;
	std::set<std::pair<int, int>> edges_nodes;

	std::regex pat{ R"((\w+/\w+/\w+)|(-?(\w+)(\.\w+)?))" };
	std::regex patf{ R"((\w+))" };

	while (std::getline(f, line)) {

		std::vector<std::string> line_tokens;

		for (std::sregex_iterator p(line.begin(), line.end(), pat); p != std::sregex_iterator{}; ++p) {
			line_tokens.push_back(p->str());
		}
		
		if (line_tokens.front() == "v") {
			Tensor node_Lpos_tensor{
				torch::tensor({std::stod(line_tokens[1]),
				std::stod(line_tokens[2]),
				std::stod(line_tokens[3])},
				opts).view({3,1}) };
			mesh.add(MeshNode(node_Lpos_tensor));
			idx_node++;
		}
		else if (line_tokens.front() == "vt") {
			Tensor node_Epos_tensor{ torch::tensor({std::stod(line_tokens[1]), std::stod(line_tokens[2])}, opts).view({2,1}) };
			mesh.add(MeshVert(node_Epos_tensor));
			idx_vert++;
		}
		else if (line_tokens.front() == "f") {
			std::array<int, 3> face_vert_token; // Three vert index of a face
			std::array<int, 3> face_node_token; // Three node index of a face
			for (int i = 1; i < line_tokens.size(); ++i) {
				std::vector<std::string> node_taken;
				for (std::sregex_iterator p(line_tokens[i].begin(), line_tokens[i].end(), patf); p != std::sregex_iterator{}; ++p) {
					node_taken.push_back(p->str());
				}
				v_vt_pair.insert(std::make_pair(std::stoi(node_taken[0]) - 1, std::stoi(node_taken[1]) - 1));
				face_vert_token[i - 1] = std::stoi(node_taken[1]) - 1; // add verts in face
				face_node_token[i - 1] = std::stoi(node_taken[0]) - 1; // add verts in face
			}
			faces_verts.push_back(face_vert_token);
			faces_nodes.push_back(face_node_token);
		}
	}
	// Connect verts and nodes
	for (auto& v_vt : v_vt_pair) {
		connect(&mesh.verts.at(v_vt.second), &mesh.nodes.at(v_vt.first));
	}
	// Collect Edges' index pairs
	for (auto& face_nodes : faces_nodes) {
		for (int idx_n0 = 0; idx_n0 < 3; ++idx_n0) {
			int idx_n1{ idx_n0 != 2 ? idx_n0 + 1 : 0 };
			if (face_nodes.at(idx_n0) < face_nodes.at(idx_n1))
				edges_nodes.insert(std::make_pair(face_nodes[idx_n0], face_nodes[idx_n1]));
			else
				edges_nodes.insert(std::make_pair(face_nodes[idx_n1], face_nodes[idx_n0]));
		}
	}
	// Add Edges
	for (auto& edge_nodes : edges_nodes) {
		mesh.add(MeshEdge(&mesh.nodes.at(edge_nodes.first), &mesh.nodes.at(edge_nodes.second)));
	}
	// Add Faces
	for (auto& face_verts : faces_verts) {
		mesh.add(MeshFace(&mesh.verts.at(face_verts[0]), &mesh.verts.at(face_verts[1]), &mesh.verts.at(face_verts[2])));
	}

	f.close();
}

void load_obj_train(const std::string& filepath, Tensor& GTPos) {
	std::ifstream obj_read;

	obj_read.open(filepath);

	std::string read_line;

	std::regex wt_sp{ "\\s+" };

	std::vector<Tensor> LPoss;

	while (std::getline(obj_read, read_line)) {
		std::vector<std::string> line_tokens;

		for (std::sregex_token_iterator p(read_line.begin(), read_line.end(), wt_sp, -1);
			p != std::sregex_token_iterator{}; ++p) {
			line_tokens.push_back(p->str());
		}

		if (line_tokens.front() == "v") {
			Tensor node_Lpos_tensor{
				torch::tensor({std::stod(line_tokens[1]),
				std::stod(line_tokens[2]),
				std::stod(line_tokens[3])},
				opts).view({3,1}) };

			LPoss.push_back(node_Lpos_tensor);
		}

	}
	obj_read.close();

	GTPos = torch::stack(LPoss, 0);
}