#pragma once
#include "CollisionUtil.h"
#include "Geometry.h"
#include "AugLag.h"

enum class ColliType { VF, EE };

template <typename T> 
inline int find(const T& x, const std::vector<T>& xs) {
	for (int i = 0; i < xs.size(); i++) 
		if (xs[i] == x) 
			return i; 
	return -1;
}

template <typename T> 
inline void remove(int i, std::vector<T>& xs) {
	xs[i] = xs.back(); 
	xs.pop_back();
}

template <typename T>
inline void exclude(const T& x, std::vector<T>& xs) {
	int i = find(x, xs); 
	if (i != -1) 
		remove(i, xs);
}

template <class T>
std::vector<T> ten2vec(Tensor a) {
	std::vector<T> ans;
	T* x = a.data_ptr<T>();
	int n = a.size(0);
	for (int i = 0; i < n; ++i)
		ans.push_back(x[i]);
	return ans;
}

struct Impact
{
	ColliType type;
	std::array<MeshNode*,4> nodes; 
	Tensor t;
	std::array<Tensor,4> ws;
	Tensor n;

	Impact() = default;
	Impact(ColliType _type, const MeshNode* _n0, const MeshNode* _n1,
		const MeshNode* _n2, const MeshNode* _n3) : type(_type) {
		nodes[0] = const_cast<MeshNode*>(_n0);
		nodes[1] = const_cast<MeshNode*>(_n1);
		nodes[2] = const_cast<MeshNode*>(_n2);
		nodes[3] = const_cast<MeshNode*>(_n3);
	}
};

struct ImpactZone {
	std::vector<MeshNode*> nodes;
	std::vector<Impact> impacts;
	std::vector<double> w, n;
	bool active;
};

struct Collision {
	Mesh* cloth_mesh, * obs_mesh, * ground_mesh;
	AccelStruct* cloth_acc, * obs_acc, *grd_acc;
	Tensor thickness;
	Tensor h;
	bool ccd;

	static const int max_iter{ 100 };

	std::vector<std::pair<MeshFace*, MeshFace*>> impact_faces;
	std::vector<Impact> impacts;

	std::vector<std::pair<MeshFace*, MeshFace*>> impact_faces_para[NThrd];
	std::vector<Impact> impacts_para[NThrd];

	Collision() = default;
	Collision(Mesh* ClothMesh, Mesh* ObstacleMesh, Mesh* GroundMesh, Tensor _thickness, Tensor _h, bool _ccd) :
		cloth_mesh(ClothMesh), obs_mesh(ObstacleMesh), ground_mesh(GroundMesh),
		cloth_acc(nullptr), obs_acc(nullptr), grd_acc(nullptr),
		thickness(_thickness), h(_h), ccd(_ccd){}

	Collision(const Collision&) = delete;

	Collision& operator=(const Collision& other) {

		if (this == &other)
			return *this;

		cloth_mesh = other.cloth_mesh;
		obs_mesh = other.obs_mesh;
		ground_mesh = other.ground_mesh;

		cloth_acc = nullptr;
		obs_acc = nullptr;
		grd_acc = nullptr;

		thickness = other.thickness;
		h = other.h;
		ccd = other.ccd;

		impact_faces = other.impact_faces;
		impacts = other.impacts;

		return *this;
	}

	bool is_free(const MeshNode* node) const;

	bool collision_response(double damp_fac);
	void update_active(const std::vector<ImpactZone*>& zones);

	void indenpendent_impacts();
	bool conflict(const Impact& imp0, const Impact& imp1);

	void add_impacts(std::vector<ImpactZone*>& zones);
	ImpactZone* find_or_create_zone(const MeshNode* node, std::vector<ImpactZone*>& zones);
	void merge_zone(ImpactZone* zone0, ImpactZone* zone1, std::vector<ImpactZone*>& zones);

	void find_impacts();
	void find_face_impacts(MeshFace* face0, MeshFace* face1);
	void comp_face_impacts(MeshFace* face0, MeshFace* face1);

	void for_overlapping_faces();
	void for_overlapping_faces(BVHNode* node);
	void for_overlapping_faces(BVHNode* node0, BVHNode* node1);

	void collect_upper_nodes(std::vector<BVHNode*>& nodes, int num_nodes_per_thread);
};

bool vf_collision_test(const MeshVert* vert, const MeshFace* face, Impact& impact);

bool ee_collision_test(const MeshEdge* edge0, const MeshEdge* edge1, Impact& impact);

bool collision_test(ColliType type, const MeshNode* node0, const MeshNode* node1, const MeshNode* node2, const MeshNode* node3, Impact& impact);

struct NormalOpt : public NLConOpt {
	ImpactZone* zone;
	Tensor inv_m;
	std::vector<double> tmp;

	NormalOpt() :zone(nullptr), inv_m(torch::zeros({}, opts)){
		nvar = 0;
		ncon = 0;
	}

	NormalOpt(ImpactZone* _zone) : zone(_zone), inv_m(torch::zeros({}, opts)) {
		nvar = zone->nodes.size()*3;
		ncon = zone->impacts.size();
		for (int n = 0; n < zone->nodes.size(); ++n)
			inv_m = inv_m + 1 / (zone->nodes[n]->m);
		inv_m = inv_m / static_cast<double>(zone->nodes.size());
		tmp = std::vector<double>(nvar);
	}

	void initialize(double* x) const override;
	void precompute(const double* x) const override;
	double objective(const double* x) const override;
	void obj_grad(const double* x, double* grad) const override;
	double constraint(const double* x, int i, int& sign, const Tensor& thickness) const override;
	void con_grad(const double* x, int i, double factor, double* grad) const override;
	void finalize(const double* x) override;
};


void precompute_derivative(real_2d_array& a, real_2d_array& q, real_2d_array& r0, std::vector<double>& lambda,
	real_1d_array& sm_1, std::vector<int>& legals, double** grads, ImpactZone* zone,
	NormalOpt& slx);

std::vector<Tensor> apply_inelastic_projection_forward(Tensor xold, Tensor ws, Tensor ns, ImpactZone* zone, const Tensor& thickness);

std::vector<Tensor> apply_inelastic_projection_backward(Tensor dldx_tn, Tensor ans_tn, Tensor q_tn, Tensor r_tn,
	Tensor lam_tn, Tensor sm1_tn, Tensor legals_tn, ImpactZone* zone);

void apply_inelastic_projection(ImpactZone* zone, const Tensor& thickness);

class InelasticProjection :public torch::autograd::Function<InelasticProjection> {
public:

	static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor xold, Tensor ws, Tensor ns, ImpactZone* zone, const Tensor& thickness) {
		std::vector<Tensor> ans = apply_inelastic_projection_forward(xold, ws, ns, zone, thickness);
		ctx->saved_data["zone_ptr"] = reinterpret_cast<intptr_t>(zone);
		ctx->save_for_backward(ans);
		return ans[0];
	}

	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list dldz) {
		auto saved = ctx->get_saved_variables();
		auto ans_tn = saved[0];
		auto q_tn = saved[1];
		auto r_tn = saved[2];
		auto lam_tn = saved[3];
		auto sm1_tn = saved[4];
		auto legals_tn = saved[5];
		ImpactZone* zone = reinterpret_cast<ImpactZone*>(ctx->saved_data["zone_ptr"].toInt());
		auto ans_back = apply_inelastic_projection_backward(dldz[0], ans_tn, q_tn, r_tn, lam_tn, sm1_tn, legals_tn, zone);
		return { ans_back[0], ans_back[1], ans_back[2], torch::empty({}).grad(), torch::empty({}).grad() };
	}
};
