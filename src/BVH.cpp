#include "BVH.h"

inline vec3f middle_xyz(const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
	vec3f ans = p1;

	for (int i = 0; i < 3; ++i)
	{
		ans[i] = 0.5 * (MIN(MIN(p1[i], p2[i]), p3[i]) + MAX(MAX(p1[i], p2[i]), p3[i]));
	}

	return ans;
}

vec3f operator+(vec3f a, vec3f b)
{
	return { a[0] + b[0],a[1] + b[1],a[2] + b[2] };
}

vec3f operator*(vec3f a, double b)
{
	return { a[0] * b,a[1] * b,a[2] * b };
}

class aap {
public:
	char _xyz;
	double _p;

	inline aap(const kDOP18& total) {
		vec3f center = total.center();
		char xyz = 2;

		if (total.width() >= total.height() && total.width() >= total.depth()) {
			xyz = 0;
		}
		else
			if (total.height() >= total.width() && total.height() >= total.depth()) {
				xyz = 1;
			}

		_xyz = xyz;
		_p = center[xyz];
	}

	inline bool inside(const vec3f& mid) const
	{
		return mid[_xyz] > _p;
	}
};

BVHTree::BVHTree(bool _ccd, Mesh& _mesh):
	ccd{ _ccd }{

	_mdl = &_mesh;

	if (!_mdl->verts.empty())
		Construct();
	else
		_root = nullptr;
}

void BVHTree::Construct()
{
	kDOP18 total;
	int count;

	int num_vtx = _mdl->verts.size(),
		num_tri = _mdl->faces.size();

	for (unsigned int i = 0; i < num_vtx; i++) {
		total += _mdl->verts[i].node->x;
		if (ccd)
			total += _mdl->verts[i].node->x_prev;
	}

	count = num_tri;

	kDOP18* tri_boxes = new kDOP18[count];
	vec3f* tri_centers = new vec3f[count];

	aap  pln(total);

	face_buffer = new MeshFace * [count];
	unsigned int left_idx = 0, right_idx = count;
	unsigned int tri_idx = 0;

	for (unsigned int i = 0; i < num_tri; i++) {
		tri_idx++;
		Tensor x;
		x = _mdl->faces[i].v[0]->node->x; vec3f p1 = dat2vec3(x.data<double>());
		x = _mdl->faces[i].v[1]->node->x; vec3f p2 = dat2vec3(x.data<double>());
		x = _mdl->faces[i].v[2]->node->x; vec3f p3 = dat2vec3(x.data<double>());
		x = _mdl->faces[i].v[0]->node->x_prev; vec3f pp1 = dat2vec3(x.data<double>());
		x = _mdl->faces[i].v[1]->node->x_prev; vec3f pp2 = dat2vec3(x.data<double>());
		x = _mdl->faces[i].v[2]->node->x_prev; vec3f pp3 = dat2vec3(x.data<double>());
		if (ccd) {

			tri_centers[tri_idx - 1] = (middle_xyz(p1, p2, p3) + middle_xyz(pp1, pp2, pp3)) * 0.5;
		}
		else {
			tri_centers[tri_idx - 1] = middle_xyz(p1, p2, p3);
		}

		if (pln.inside(tri_centers[tri_idx - 1]))
			face_buffer[left_idx++] = &_mdl->faces[i];
		else
			face_buffer[--right_idx] = &_mdl->faces[i];

		tri_boxes[tri_idx - 1] += p1;
		tri_boxes[tri_idx - 1] += p2;
		tri_boxes[tri_idx - 1] += p3;

		if (ccd) {
			tri_boxes[tri_idx - 1] += pp1;
			tri_boxes[tri_idx - 1] += pp2;
			tri_boxes[tri_idx - 1] += pp3;
		}
	}

	_root = new BVHNode();
	_root->_box = total;
	//_root->_count = count;

	if (count == 1) {
		_root->_face = &_mdl->faces[0];
		_root->_left = _root->_right = NULL;
	}
	else {
		if (left_idx == 0 || left_idx == count)
			left_idx = count / 2;

		_root->_left = new BVHNode(_root, face_buffer, left_idx, tri_boxes, tri_centers);
		_root->_right = new BVHNode(_root, face_buffer + left_idx, count - left_idx, tri_boxes, tri_centers);
	}

	delete[] tri_boxes;
	delete[] tri_centers;

}

BVHTree::~BVHTree() {
	if (!_root)
		return;
	delete _root;
}

void BVHTree::refit() {
	_root->refit(ccd);
}

void BVHNode::refit(bool _ccd) {
	if (isLeaf())
		_box = face_box(_face, _ccd);
	else {
		_left->refit(_ccd);
		_right->refit(_ccd);

		_box = _left->_box + _right->_box;
	}
}

// called by root
BVHNode::BVHNode() {
	_face = nullptr;
	_left = nullptr;
	_right = nullptr;
	_parent = nullptr;
	_actived = true;
}

// called by leaf
BVHNode::BVHNode(BVHNode* parent, MeshFace* face, kDOP18* face_boxes, vec3f* face_centers) {
	_left = nullptr;
	_right = nullptr;
	_parent = parent;
	_face = face;
	_box = face_boxes[_face->index];
	_actived = true;
}

// called by nodes
BVHNode::BVHNode(BVHNode* parent, MeshFace** lst, int lst_num, kDOP18* tri_boxes, vec3f* tri_centers) {

	assert(lst_num > 0);
	_left = _right = NULL;
	_parent = parent;
	_face = NULL;
	//_count = lst_num;
	_actived = true;

	if (lst_num == 1) {
		_face = lst[0];
		_box = tri_boxes[lst[0]->index];
	}
	else { // try to split them
		for (unsigned int t = 0; t < lst_num; t++) {
			int i = lst[t]->index;
			_box += tri_boxes[i];
		}

		if (lst_num == 2) { // must split it!
			_left = new BVHNode(this, lst[0], tri_boxes, tri_centers);
			_right = new BVHNode(this, lst[1], tri_boxes, tri_centers);
		}
		else {
			aap pln(_box);
			unsigned int left_idx = 0, right_idx = lst_num - 1;

			for (unsigned int t = 0; t < lst_num; t++) {
				int i = lst[left_idx]->index;
				if (pln.inside(tri_centers[i]))
					left_idx++;
				else {// swap it
					MeshFace* tmp = lst[left_idx];
					lst[left_idx] = lst[right_idx];
					lst[right_idx--] = tmp;
				}
			}

			int hal = lst_num / 2;
			if (left_idx == 0 || left_idx == lst_num)
			{
				_left = new BVHNode(this, lst, hal, tri_boxes, tri_centers);
				_right = new BVHNode(this, lst + hal, lst_num - hal, tri_boxes, tri_centers);

			}
			else {
				_left = new BVHNode(this, lst, left_idx, tri_boxes, tri_centers);
				_right = new BVHNode(this, lst + left_idx, lst_num - left_idx, tri_boxes, tri_centers);
			}

		}
	}
}

BVHNode::~BVHNode() {
	if (_left) 
		delete _left;	
	if (_right) 
		delete _right;	
}

kDOP18 vert_box(const MeshVert* vert, bool ccd) {
	return node_box(vert->node, ccd);
}

kDOP18 node_box(const MeshNode* node, bool ccd) {
	auto x = node->x;

	kDOP18 box(dat2vec3(x.data<double>()));

	if (ccd)
	{
		box += node->x_prev;
	}

	return box;
}

kDOP18 face_box(const MeshFace* face, bool ccd) {
	kDOP18 box = vert_box(face->v[0], ccd);
	for (int v = 1; v < 3; ++v) {
		box += vert_box(face->v[v], ccd);
	}
	return box;
}

kDOP18 dilate(const kDOP18& box, double d) {

	static double sqrt2 = std::sqrt(2.0);

	kDOP18 dbox = box;

	for (int i = 0; i < 3; i++)
	{
		dbox._dist[i] -= d;
		dbox._dist[i + 9] += d;
	}

	for (int i = 0; i < 6; i++)
	{
		dbox._dist[3 + i] -= sqrt2 * d;
		dbox._dist[3 + i + 9] += sqrt2 * d;
	}

	return dbox;
}

bool overlap(const kDOP18& box0, const kDOP18& box1, Tensor thickness) {
	return box0.overlaps(dilate(box1, thickness.item<double>()));
}