#include "Saver.h"

void SaveTrainProc(H5File& file, const int& curr_step, const int& train_steps, const Tensor& loss, const torch::OrderedDict<std::string, Tensor>& paras) {

	const static int RANK_TRAIN{ 2 };

	if (curr_step == 0) {
		hsize_t dimsl[2], dimsp[2];

		dimsl[0] = train_steps;
		dimsl[1] = 1;

		dimsp[0] = train_steps;

		DataSpace dataspaceL{ RANK_TRAIN, dimsl };

		DataSet datasetL{ file.createDataSet("Loss", PredType::NATIVE_DOUBLE, dataspaceL) };

		for (auto const& para : paras) {
			dimsp[1] = para.value().view({ -1 }).size(0);
			DataSpace dataspaceP{ RANK_TRAIN, dimsp };
			file.createDataSet(para.key(), PredType::NATIVE_DOUBLE, dataspaceP);
		}
	}

	hsize_t start[2], stride[2], count[2], block[2];
	hsize_t startm[2], stridem[2], countm[2], blockm[2];

	start[0] = curr_step; start[1] = 0;
	stride[0] = 1; stride[1] = 1;
	count[0] = 1; count[1] = 1;
	block[0] = 1; block[1] = 1;

	startm[0] = 0; startm[1] = 0;
	stridem[0] = 1; stridem[1] = 1;
	countm[0] = 1; countm[1] = 1;
	blockm[0] = 1; blockm[1] = 1;

	double loss_ary[1][1];
	double loss_pra[1][1];
	double loss_G[1][3];

	loss_ary[0][0] = loss.item<double>();

	DataSet datasetL{ file.openDataSet("Loss") };
	DataSpace dataspaceL{ datasetL.getSpace() };

	dataspaceL.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

	hsize_t dimm[2];

	dimm[0] = 1, dimm[1] = 1;

	DataSpace spacelm(2, dimm);

	spacelm.selectHyperslab(H5S_SELECT_SET, countm, startm, stridem, blockm);

	datasetL.write(loss_ary, PredType::NATIVE_DOUBLE, spacelm, dataspaceL);

	for (auto const& para : paras) {
		bool is_G{ false };
		if (para.key() == "G") {
			is_G = true;
			for (int i = 0; i < 3; ++i) {
				loss_G[0][i] = para.value()[i].item<double>();
			}
			dimm[1] = 3;
			block[1] = 3;
			countm[1] = 3;
		}
		else {
			is_G = false;
			loss_pra[0][0] = para.value().item<double>();
			dimm[1] = 1;
			block[1] = 1;
			countm[1] = 1;
		}

		DataSet datasetP{ file.openDataSet(para.key()) };
		DataSpace dataspaceP{ datasetP.getSpace() };
		dataspaceP.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

		DataSpace spacePm{ RANK_TRAIN , dimm };
		spacePm.selectHyperslab(H5S_SELECT_SET, countm, startm, stridem, blockm);

		if (is_G)
			datasetP.write(loss_G, PredType::NATIVE_DOUBLE, spacePm, dataspaceP);
		else
			datasetP.write(loss_pra, PredType::NATIVE_DOUBLE, spacePm, dataspaceP);
	}
}

void SaveTrainProc(H5File& file, const int& curr_step, const int& train_steps, const Tensor& loss, const std::map<std::string, Tensor>& paras) {
	// Saver for constrained parameters
	const static int RANK_TRAIN{ 2 };

	if (curr_step == 0) {
		hsize_t dimsl[2], dimsp[2];

		dimsl[0] = train_steps;
		dimsl[1] = 1;

		dimsp[0] = train_steps;

		DataSpace dataspaceL{ RANK_TRAIN, dimsl };

		DataSet datasetL{ file.createDataSet("Loss", PredType::NATIVE_DOUBLE, dataspaceL) };

		for (auto const& para : paras) {
			dimsp[1] = para.second.view({ -1 }).size(0);
			DataSpace dataspaceP{ RANK_TRAIN, dimsp };
			file.createDataSet(para.first, PredType::NATIVE_DOUBLE, dataspaceP);
		}
	}

	hsize_t start[2], stride[2], count[2], block[2];
	hsize_t startm[2], stridem[2], countm[2], blockm[2];

	start[0] = curr_step; start[1] = 0;
	stride[0] = 1; stride[1] = 1;
	count[0] = 1; count[1] = 1;
	block[0] = 1; block[1] = 1;

	startm[0] = 0; startm[1] = 0;
	stridem[0] = 1; stridem[1] = 1;
	countm[0] = 1; countm[1] = 1;
	blockm[0] = 1; blockm[1] = 1;

	double loss_ary[1][1];
	double loss_pra[1][1];
	double loss_G[1][3];

	loss_ary[0][0] = loss.item<double>();

	DataSet datasetL{ file.openDataSet("Loss") };
	DataSpace dataspaceL{ datasetL.getSpace() };

	dataspaceL.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

	hsize_t dimm[2];

	dimm[0] = 1, dimm[1] = 1;

	DataSpace spacelm(2, dimm);

	spacelm.selectHyperslab(H5S_SELECT_SET, countm, startm, stridem, blockm);

	datasetL.write(loss_ary, PredType::NATIVE_DOUBLE, spacelm, dataspaceL);

	for (auto const& para : paras) {
		bool is_G{ false };
		if (para.first == "G") {
			is_G = true;
			for (int i = 0; i < 3; ++i) {
				loss_G[0][i] = para.second[i].item<double>();
			}
			dimm[1] = 3;
			block[1] = 3;
			countm[1] = 3;
		}
		else {
			is_G = false;
			loss_pra[0][0] = para.second.item<double>();
			dimm[1] = 1;
			block[1] = 1;
			countm[1] = 1;
		}

		DataSet datasetP{ file.openDataSet(para.first) };
		DataSpace dataspaceP{ datasetP.getSpace() };
		dataspaceP.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

		DataSpace spacePm{ RANK_TRAIN , dimm };
		spacePm.selectHyperslab(H5S_SELECT_SET, countm, startm, stridem, blockm);

		if (is_G)
			datasetP.write(loss_G, PredType::NATIVE_DOUBLE, spacePm, dataspaceP);
		else
			datasetP.write(loss_pra, PredType::NATIVE_DOUBLE, spacePm, dataspaceP);
	}

}

void SaveClothHomoYarn(const Cloth& cloth, const Environment& env, std::shared_ptr<H5File> file, int step_num, int step)
{
	static const int DIM_PROP{ 1 };
	static const int DIM_LPOS{ 3 };
	static const int DIM_EPOS{ 2 };
	static const int RANK_PROP{ 1 };
	static const int RANK_SIM{ 2 };

	if (step == 0) {
		H5ClothSize dset_ClothSize[DIM_PROP];
		H5Physical dset_Physical[DIM_PROP];

		dset_ClothSize[0] = H5ClothSize{ cloth.width, cloth.length };
		dset_Physical[0] = H5Physical{ env.G[2].item<double>(), cloth.yarns[0].rho.item<double>(),
			cloth.R, cloth.L, cloth.yarns[0].Y.item<double>(), cloth.yarns[0].B.item<double>(),
			cloth.kc.item<double>(), cloth.kf.item<double>(),
			cloth.df.item<double>(), cloth.mu.item<double>(),
			cloth.S.item<double>(), cloth.handle_stiffness.item<double>(),
			env.wind.WindVelocity[0].item<double>(), env.wind.WindVelocity[1].item<double>(),
			env.wind.WindVelocity[2].item<double>(), env.wind.WindDensity.item<double>(),
			env.wind.WindDrag.item<double>(), static_cast<int>(cloth.woven_pattern) };

		Group group_prop(file->createGroup("/Properties"));

		hsize_t dim[] = { DIM_PROP };
		DataSpace dataspace(RANK_PROP, dim);

		CompType cloth_size(sizeof(H5ClothSize));
		cloth_size.insertMember("width", HOFFSET(H5ClothSize, width), PredType::NATIVE_INT);
		cloth_size.insertMember("length", HOFFSET(H5ClothSize, length), PredType::NATIVE_INT);

		std::unique_ptr<DataSet> dataset{ new DataSet(group_prop.createDataSet("Size", cloth_size, dataspace)) };
		dataset->write(dset_ClothSize, cloth_size);

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

		dataset = std::make_unique<DataSet>(group_prop.createDataSet("Physical properties", cloth_physical, dataspace));
		dataset->write(dset_Physical, cloth_physical);

		dataset->close();

		hsize_t sim_L_dims[RANK_SIM], sim_E_dims[RANK_SIM];
		sim_L_dims[0] = step_num;
		sim_L_dims[1] = DIM_LPOS;
		sim_E_dims[0] = step_num;
		sim_E_dims[1] = DIM_EPOS;
		DataSpace SimLDataspace(RANK_SIM, sim_L_dims);
		DataSpace SimEDataspace(RANK_SIM, sim_E_dims);

		Group group_sim(file->createGroup("/Steps"));
		Group group_pos(group_sim.createGroup("Position"));
		Group group_vel(group_sim.createGroup("Velocity"));

		double fillvalue{ 0.0 };
		DSetCreatPropList plist;
		plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

		for (int v = 0; v < cloth.length; ++v)
			for (int u = 0; u < cloth.width; ++u) {
				H5std_string group_name = "Node " + std::to_string(u) + " " + std::to_string(v);

				Group pos_temp = group_pos.createGroup(group_name);
				Group vel_temp = group_vel.createGroup(group_name);

				pos_temp.createDataSet("LPOS", PredType::NATIVE_DOUBLE, SimLDataspace, plist);
				pos_temp.createDataSet("LPOSFix", PredType::NATIVE_DOUBLE, SimLDataspace, plist);
				vel_temp.createDataSet("LVEL", PredType::NATIVE_DOUBLE, SimLDataspace, plist);
				pos_temp.createDataSet("EPOS", PredType::NATIVE_DOUBLE, SimEDataspace, plist);
				vel_temp.createDataSet("EVEL", PredType::NATIVE_DOUBLE, SimEDataspace, plist);
				pos_temp.createDataSet("EPOSBar", PredType::NATIVE_DOUBLE, SimEDataspace, plist);
			}
	}

	for (int v = 0; v < cloth.length; ++v)
		for (int u = 0; u < cloth.width; ++u)
		{
			std::string LPosDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/LPOS";
			std::string LVelDatasetName = "/Steps/Velocity/Node " + std::to_string(u) + " " + std::to_string(v) + "/LVEL";
			std::string EPosDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/EPOS";
			std::string EVelDatasetName = "/Steps/Velocity/Node " + std::to_string(u) + " " + std::to_string(v) + "/EVEL";
			std::string LPosFixDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/LPOSFix";
			std::string EPosBarDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/EPOSBar";

			std::string idx = std::to_string(u) + "," + std::to_string(v);

			double* LPos_ptr = cloth.nodes.at(idx).LPos.data_ptr<double>();
			double* LVel_ptr = cloth.nodes.at(idx).LVel.data_ptr<double>();
			double* EPos_ptr = cloth.nodes.at(idx).EPos.data_ptr<double>();
			double* EVel_ptr = cloth.nodes.at(idx).EVel.data_ptr<double>();
			double* LPosFix_ptr = cloth.nodes.at(idx).LPosFix.data_ptr<double>();
			double* EPosBar_ptr = cloth.nodes.at(idx).EPosBar.data_ptr<double>();

			double NodeLPos[1][3];
			double NodeLVel[1][3];
			double NodeEPos[1][2];
			double NodeEVel[1][2];
			double NodeLPosFix[1][3];
			double NodeEPosBar[1][2];

			for (int i = 0; i < 3; ++i) {
				NodeLPos[0][i] = *(LPos_ptr++);
				NodeLVel[0][i] = *(LVel_ptr++);
				NodeLPosFix[0][i] = *(LPosFix_ptr++);
			}

			for (int i = 0; i < 2; ++i) {
				NodeEPos[0][i] = *(EPos_ptr++);
				NodeEVel[0][i] = *(EVel_ptr++);
				NodeEPosBar[0][i] = *(EPosBar_ptr++);
			}

			// Fill L Position
			std::unique_ptr<DataSet> dataset = std::make_unique<DataSet>(file->openDataSet(LPosDatasetName));
			DataSpace Lpos_fspace = dataset->getSpace();

			hsize_t start[2], stride[2], count[2], block[2];
			hsize_t mstart[2], mstride[2], mcount[2], mblock[2];

			start[0] = step;
			start[1] = 0;
			stride[0] = 1;
			stride[1] = 3;
			count[0] = 1;
			count[1] = 1;
			block[0] = 1;
			block[1] = 3;

			Lpos_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			mstart[0] = 0;
			mstart[1] = 0;
			mstride[0] = 1;
			mstride[1] = 1;
			mcount[0] = 1;
			mcount[1] = 3;
			mblock[0] = 1;
			mblock[1] = 1;

			hsize_t dimm[2];
			dimm[0] = 1;
			dimm[1] = 3;

			DataSpace Lpos_mspace{ RANK_SIM, dimm };
			Lpos_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeLPos, PredType::NATIVE_DOUBLE, Lpos_mspace, Lpos_fspace);

			// Fill L Position Fix
			dataset = std::make_unique<DataSet>(file->openDataSet(LPosFixDatasetName));
			DataSpace Lposfix_fspace = dataset->getSpace();

			Lposfix_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			DataSpace Lposfix_mspace{ RANK_SIM, dimm };
			Lposfix_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeLPosFix, PredType::NATIVE_DOUBLE, Lpos_mspace, Lpos_fspace);

			// Fill L Velocity
			dataset = std::make_unique<DataSet>(file->openDataSet(LVelDatasetName));
			DataSpace Lvel_fspace = dataset->getSpace();

			Lvel_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			DataSpace Lvel_mspace{ RANK_SIM, dimm };
			Lvel_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeLVel, PredType::NATIVE_DOUBLE, Lvel_mspace, Lvel_fspace);

			// Fill E Position
			dataset = std::make_unique<DataSet>(file->openDataSet(EPosDatasetName));
			DataSpace Epos_fspace = dataset->getSpace();

			start[0] = step;
			start[1] = 0;
			stride[0] = 1;
			stride[1] = 2;
			count[0] = 1;
			count[1] = 1;
			block[0] = 1;
			block[1] = 2;

			Epos_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			mstart[0] = 0;
			mstart[1] = 0;
			mstride[0] = 1;
			mstride[1] = 1;
			mcount[0] = 1;
			mcount[1] = 2;
			mblock[0] = 1;
			mblock[1] = 1;

			dimm[0] = 1;
			dimm[1] = 2;

			DataSpace Epos_mspace{ RANK_SIM, dimm };
			Epos_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeEPos, PredType::NATIVE_DOUBLE, Epos_mspace, Epos_fspace);

			// Fill E Velocity
			dataset = std::make_unique<DataSet>(file->openDataSet(EVelDatasetName));
			DataSpace Evel_fspace = dataset->getSpace();

			Evel_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			DataSpace Evel_mspace{ RANK_SIM, dimm };
			Evel_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeEVel, PredType::NATIVE_DOUBLE, Evel_mspace, Evel_fspace);

			// Fill E Position Anchor
			dataset = std::make_unique<DataSet>(file->openDataSet(EPosBarDatasetName));
			DataSpace Eposbar_fspace = dataset->getSpace();

			Eposbar_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			DataSpace Eposbar_mspace{ RANK_SIM, dimm };
			Eposbar_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeEPosBar, PredType::NATIVE_DOUBLE, Eposbar_mspace, Eposbar_fspace);
		}
}

void SaveClothHeterYarn(const Cloth& cloth, const Environment& env, std::shared_ptr<H5File> file, int step_num, int step)
{
	static const int DIM_PROP{ 1 };
	static const int DIM_LPOS{ 3 };
	static const int DIM_EPOS{ 2 };
	static const int RANK_PROP{ 1 };
	static const int RANK_SIM{ 2 };

	if (step == 0) {
		H5ClothSize dset_ClothSize[DIM_PROP];
		H5PhysicalHeterYarn dset_PhysicalHeterYarn[DIM_PROP];

		dset_ClothSize[0] = H5ClothSize{ cloth.width, cloth.length };
		dset_PhysicalHeterYarn[0] = H5PhysicalHeterYarn{ env.G[2].item<double>(),
			cloth.yarns[0].rho.item<double>(), cloth.yarns[1].rho.item<double>(),
			cloth.R, cloth.L, cloth.yarns[0].Y.item<double>(), cloth.yarns[1].Y.item<double>(),
			cloth.yarns[0].B.item<double>(), cloth.yarns[1].B.item<double>(),
			cloth.kc.item<double>(), cloth.kf.item<double>(),
			cloth.df.item<double>(), cloth.mu.item<double>(),
			cloth.S.item<double>(), cloth.handle_stiffness.item<double>(),
			env.wind.WindVelocity[0].item<double>(), env.wind.WindVelocity[1].item<double>(),
			env.wind.WindVelocity[2].item<double>(), env.wind.WindDensity.item<double>(),
			env.wind.WindDrag.item<double>(), static_cast<int>(cloth.woven_pattern) };

		Group group_prop(file->createGroup("/Properties"));

		hsize_t dim[] = { DIM_PROP };
		DataSpace dataspace(RANK_PROP, dim);

		CompType cloth_size(sizeof(H5ClothSize));
		cloth_size.insertMember("width", HOFFSET(H5ClothSize, width), PredType::NATIVE_INT);
		cloth_size.insertMember("length", HOFFSET(H5ClothSize, length), PredType::NATIVE_INT);

		std::unique_ptr<DataSet> dataset{ new DataSet(group_prop.createDataSet("Size", cloth_size, dataspace)) };
		dataset->write(dset_ClothSize, cloth_size);

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

		dataset = std::make_unique<DataSet>(group_prop.createDataSet("Physical properties", cloth_physical, dataspace));
		dataset->write(dset_PhysicalHeterYarn, cloth_physical);

		dataset->close();

		hsize_t sim_L_dims[RANK_SIM], sim_E_dims[RANK_SIM];
		sim_L_dims[0] = step_num;
		sim_L_dims[1] = DIM_LPOS;
		sim_E_dims[0] = step_num;
		sim_E_dims[1] = DIM_EPOS;
		DataSpace SimLDataspace(RANK_SIM, sim_L_dims);
		DataSpace SimEDataspace(RANK_SIM, sim_E_dims);

		Group group_sim(file->createGroup("/Steps"));
		Group group_pos(group_sim.createGroup("Position"));
		Group group_vel(group_sim.createGroup("Velocity"));

		double fillvalue{ 0.0 };
		DSetCreatPropList plist;
		plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);

		for (int v = 0; v < cloth.length; ++v)
			for (int u = 0; u < cloth.width; ++u) {
				H5std_string group_name = "Node " + std::to_string(u) + " " + std::to_string(v);

				Group pos_temp = group_pos.createGroup(group_name);
				Group vel_temp = group_vel.createGroup(group_name);

				pos_temp.createDataSet("LPOS", PredType::NATIVE_DOUBLE, SimLDataspace, plist);
				pos_temp.createDataSet("LPOSFix", PredType::NATIVE_DOUBLE, SimLDataspace, plist);
				vel_temp.createDataSet("LVEL", PredType::NATIVE_DOUBLE, SimLDataspace, plist);
				pos_temp.createDataSet("EPOS", PredType::NATIVE_DOUBLE, SimEDataspace, plist);
				vel_temp.createDataSet("EVEL", PredType::NATIVE_DOUBLE, SimEDataspace, plist);
				pos_temp.createDataSet("EPOSBar", PredType::NATIVE_DOUBLE, SimEDataspace, plist);
			}
	}

	for (int v = 0; v < cloth.length; ++v)
		for (int u = 0; u < cloth.width; ++u)
		{
			std::string LPosDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/LPOS";
			std::string LVelDatasetName = "/Steps/Velocity/Node " + std::to_string(u) + " " + std::to_string(v) + "/LVEL";
			std::string EPosDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/EPOS";
			std::string EVelDatasetName = "/Steps/Velocity/Node " + std::to_string(u) + " " + std::to_string(v) + "/EVEL";
			std::string LPosFixDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/LPOSFix";
			std::string EPosBarDatasetName = "/Steps/Position/Node " + std::to_string(u) + " " + std::to_string(v) + "/EPOSBar";

			std::string idx = std::to_string(u) + "," + std::to_string(v);

			double* LPos_ptr = cloth.nodes.at(idx).LPos.data_ptr<double>();
			double* LVel_ptr = cloth.nodes.at(idx).LVel.data_ptr<double>();
			double* EPos_ptr = cloth.nodes.at(idx).EPos.data_ptr<double>();
			double* EVel_ptr = cloth.nodes.at(idx).EVel.data_ptr<double>();
			double* LPosFix_ptr = cloth.nodes.at(idx).LPosFix.data_ptr<double>();
			double* EPosBar_ptr = cloth.nodes.at(idx).EPosBar.data_ptr<double>();

			double NodeLPos[1][3];
			double NodeLVel[1][3];
			double NodeEPos[1][2];
			double NodeEVel[1][2];
			double NodeLPosFix[1][3];
			double NodeEPosBar[1][2];

			for (int i = 0; i < 3; ++i) {
				NodeLPos[0][i] = *(LPos_ptr++);
				NodeLVel[0][i] = *(LVel_ptr++);
				NodeLPosFix[0][i] = *(LPosFix_ptr++);
			}

			for (int i = 0; i < 2; ++i) {
				NodeEPos[0][i] = *(EPos_ptr++);
				NodeEVel[0][i] = *(EVel_ptr++);
				NodeEPosBar[0][i] = *(EPosBar_ptr++);
			}

			// Fill L Position
			std::unique_ptr<DataSet> dataset = std::make_unique<DataSet>(file->openDataSet(LPosDatasetName));
			DataSpace Lpos_fspace = dataset->getSpace();

			hsize_t start[2], stride[2], count[2], block[2];
			hsize_t mstart[2], mstride[2], mcount[2], mblock[2];

			start[0] = step;
			start[1] = 0;
			stride[0] = 1;
			stride[1] = 3;
			count[0] = 1;
			count[1] = 1;
			block[0] = 1;
			block[1] = 3;

			Lpos_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			mstart[0] = 0;
			mstart[1] = 0;
			mstride[0] = 1;
			mstride[1] = 1;
			mcount[0] = 1;
			mcount[1] = 3;
			mblock[0] = 1;
			mblock[1] = 1;

			hsize_t dimm[2];
			dimm[0] = 1;
			dimm[1] = 3;

			DataSpace Lpos_mspace{ RANK_SIM, dimm };
			Lpos_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeLPos, PredType::NATIVE_DOUBLE, Lpos_mspace, Lpos_fspace);

			// Fill L Position Fix
			dataset = std::make_unique<DataSet>(file->openDataSet(LPosFixDatasetName));
			DataSpace Lposfix_fspace = dataset->getSpace();

			Lposfix_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			DataSpace Lposfix_mspace{ RANK_SIM, dimm };
			Lposfix_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeLPosFix, PredType::NATIVE_DOUBLE, Lpos_mspace, Lpos_fspace);

			// Fill L Velocity
			dataset = std::make_unique<DataSet>(file->openDataSet(LVelDatasetName));
			DataSpace Lvel_fspace = dataset->getSpace();

			Lvel_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			DataSpace Lvel_mspace{ RANK_SIM, dimm };
			Lvel_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeLVel, PredType::NATIVE_DOUBLE, Lvel_mspace, Lvel_fspace);

			// Fill E Position
			dataset = std::make_unique<DataSet>(file->openDataSet(EPosDatasetName));
			DataSpace Epos_fspace = dataset->getSpace();

			start[0] = step;
			start[1] = 0;
			stride[0] = 1;
			stride[1] = 2;
			count[0] = 1;
			count[1] = 1;
			block[0] = 1;
			block[1] = 2;

			Epos_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			mstart[0] = 0;
			mstart[1] = 0;
			mstride[0] = 1;
			mstride[1] = 1;
			mcount[0] = 1;
			mcount[1] = 2;
			mblock[0] = 1;
			mblock[1] = 1;

			dimm[0] = 1;
			dimm[1] = 2;

			DataSpace Epos_mspace{ RANK_SIM, dimm };
			Epos_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeEPos, PredType::NATIVE_DOUBLE, Epos_mspace, Epos_fspace);

			// Fill E Velocity
			dataset = std::make_unique<DataSet>(file->openDataSet(EVelDatasetName));
			DataSpace Evel_fspace = dataset->getSpace();

			Evel_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			DataSpace Evel_mspace{ RANK_SIM, dimm };
			Evel_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeEVel, PredType::NATIVE_DOUBLE, Evel_mspace, Evel_fspace);

			// Fill E Position Anchor
			dataset = std::make_unique<DataSet>(file->openDataSet(EPosBarDatasetName));
			DataSpace Eposbar_fspace = dataset->getSpace();

			Eposbar_fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);

			DataSpace Eposbar_mspace{ RANK_SIM, dimm };
			Eposbar_mspace.selectHyperslab(H5S_SELECT_SET, mcount, mstart, mstride, mblock);

			dataset->write(NodeEPosBar, PredType::NATIVE_DOUBLE, Eposbar_mspace, Eposbar_fspace);
		}
}