#include "Obstacle.h"

void Obstacle::compute_mesh_mass() {
	obsMesh.compute_ms_data();
	obsMesh.compute_ws_data();

	for (int n = 0; n < obsMesh.nodes.size(); ++n) {
		obsMesh.nodes[n].m = torch::zeros({ 1 }, opts);
	}
	for (int f = 0; f < obsMesh.faces.size(); ++f) {
		obsMesh.faces[f].m = obsMesh.faces[f].a * density;
		for (int v = 0; v < 3; ++v) {
			obsMesh.faces[f].v[v]->node->m = obsMesh.faces[f].v[v]->node->m + obsMesh.faces[f].m / 3.0;
		}
	}
}