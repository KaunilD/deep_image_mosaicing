#ifndef KEYPOINT_H
#define KEYPOINT_H

struct Keypoint {
	int row, col;
	float conf;
	int id;
	
	Keypoint(int r, int c, int v) : row(r), col(c), conf(v), id(0) {};
	
	void setId(int t_id) { id = t_id; }
};

#endif KEYPOINT_H