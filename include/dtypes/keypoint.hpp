#ifndef KEYPOINT_H
#define KEYPOINT_H

struct Keypoint {
	Eigen::Vector2f m_loc;
	float m_conf;
	int id;
	
	Keypoint(int r, int c, int v) : m_loc(r, c), m_conf(v), id(0) {};
	
	Keypoint() {
		m_loc = Eigen::Vector2f(0, 0);
		m_conf = 0.0f;
	}

	Keypoint(Eigen::Vector2f loc, float conf) : m_loc(loc), m_conf(conf) {};

	void setId(int t_id) { id = t_id; }

};

#endif KEYPOINT_H