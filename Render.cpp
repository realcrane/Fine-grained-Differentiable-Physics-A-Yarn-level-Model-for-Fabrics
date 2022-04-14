#include "Render.h"

void RenderCloth(Cloth& cloth, std::string save_path, RenderType type)
{
	std::ofstream f;
	f.open(save_path);
	f.precision(5);

	static double unit_uv{ 1.0 / (std::max(cloth.width, cloth.length) - 1) };

	f << "o cloth\n";

	if (type == RenderType::MeshTriangle) {
		// obj v
		for (int i = 0; i < cloth.length; ++i)
			for (int j = 0; j < cloth.width; ++j) {
				std::string idx = std::to_string(j) + "," + std::to_string(i);
				f << "v ";
				for (int xyz = 0; xyz < 3; ++xyz)
					f << std::fixed << cloth.nodes.at(idx).LPos[xyz].item<double>() << " ";
				f << "\n";
			}
		// obj vt 
		for (int i = 0; i < cloth.length; ++i)
			for (int j = 0; j < cloth.width; ++j) {
				std::string idx = std::to_string(j) + "," + std::to_string(i);
				f << "vt ";
				f << std::fixed << std::to_string(j * unit_uv) << " " << std::to_string(i * unit_uv) << "\n";
			}
		// obj vn
		for (int i = 0; i < cloth.length; ++i)
			for (int j = 0; j < cloth.width; ++j) {
				std::string idx = std::to_string(j) + "," + std::to_string(i);
				Tensor vn = ZERO31;
				for (auto& face : cloth.nodes.at(idx).adj_faces)
					vn = vn + face->n;	
				vn = vn / static_cast<int>(cloth.nodes.at(idx).adj_faces.size());
				f << "vn ";
				for (int xyz = 0; xyz < 3; ++xyz)
					f << std::fixed << vn[xyz].item<double>() << " ";
				f << "\n";
			}
		// obj f
		for (int i = 0; i < cloth.length - 1; ++i)
			for (int j = 0; j < cloth.width - 1; ++j) {
				std::string top_left{ std::to_string(j + i * cloth.width + 1) },
					top_right{ std::to_string(j + 1 + i * cloth.width + 1) },
					bot_left{ std::to_string(j + (i + 1) * cloth.width + 1) },
					bot_right{ std::to_string(j + 1 + (i + 1) * cloth.width + 1) };
				if ((i + j) % 2 == 0) {
					f << "f "
						<< top_left << "/" << top_left << "/" << top_left << " "
						<< top_right << "/" << top_right << "/" << top_right << " "
						<< bot_right << "/" << bot_right << "/" << bot_right << "\n";

					f << "f "
						<< top_left << "/" << top_left << "/" << top_left << " "
						<< bot_right << "/" << bot_right << "/" << bot_right << " "
						<< bot_left << "/" << bot_left << "/" << bot_left << "\n";
				}
				else {
					f << "f "
						<< top_left << "/" << top_left << "/" << top_left << " "
						<< top_right << "/" << top_right << "/" << top_right << " "
						<< bot_left << "/" << bot_left << "/" << bot_left << "\n";

					f << "f "
						<< top_right << "/" << top_right << "/" << top_right << " "
						<< bot_right << "/" << bot_right << "/" << bot_right << " "
						<< bot_left << "/" << bot_left << "/" << bot_left << "\n";
				}
			}
	}
	else if (type == RenderType::MeshRectangle) {
		// obj v
		for (int i = 0; i < cloth.length; ++i)
			for (int j = 0; j < cloth.width; ++j) {
				std::string idx = std::to_string(j) + "," + std::to_string(i);
				f << "v ";
				for (int xyz = 0; xyz < 3; ++xyz)
					f << std::fixed << cloth.nodes.at(idx).LPos[xyz].item<double>() << " ";
				f << "\n";
			}
		for (int i = 0; i < cloth.length - 1; ++i)
			for (int j = 0; j < cloth.width - 1; ++j) {
				std::string top_left{ std::to_string(j + i * cloth.width + 1) },
					top_right{ std::to_string(j + 1 + i * cloth.width + 1) },
					bot_left{ std::to_string(j + (i + 1) * cloth.width + 1) },
					bot_right{ std::to_string(j + 1 + (i + 1) * cloth.width + 1) };

				f << "f " << top_left << " " << top_right << " " << bot_right << " " << bot_left << "\n";
			}
	}
	else {
		// Warp nodes v
		for (int v = 1; v < cloth.length - 1; ++v)
			for (int u = 0; u < cloth.width; ++u) {
				f << "v ";
				std::string node_idx{ std::to_string(u) + "," + std::to_string(v) };
				for (int co = 0; co < 3; ++co)
					f << std::fixed << std::setprecision(5) << cloth.warp_nodes.at(node_idx).LPos[co].item<double>() << " ";
				f << "\n";
			}
		// Weft nodes v
		for (int u = 1; u < cloth.width - 1; ++u)
			for (int v = 0; v < cloth.length; ++v) {
				f << "v ";
				std::string node_idx{ std::to_string(u) + "," + std::to_string(v) };
				for (int co = 0; co < 3; ++co)
					f << std::fixed << std::setprecision(5) << cloth.weft_nodes.at(node_idx).LPos[co].item<double>() << " ";
				f << "\n";
			}
		// Warp splines l
		for (int v = 1; v < cloth.length - 1; ++v)
			for (int u = 0; u < cloth.width - 1; ++u) {
				f << "l " <<
					(v - 1) * cloth.width + u + 1 << " " <<
					(v - 1) * cloth.width + u + 2 <<
					" \n";
			}
		// Weft splines l
		for (int u = 1; u < cloth.width - 1; ++u)
			for (int v = 0; v < cloth.length - 1; ++v) {
				f << "l " <<
					(cloth.length - 2) * cloth.width + (u - 1) * cloth.length + v + 1 << " " <<
					(cloth.length - 2) * cloth.width + (u - 1) * cloth.length + v + 2 <<
					" \n";
			}
	}

	f.close();
}