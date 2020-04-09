#ifndef MATCHER_H
#define MATCHER_H

#include "libs.hpp"
#include "features.hpp"
#include "math.h"
class Matcher {
public:
	enum MType {
		BFMatcher,
		FLANNMatcher
	};

	Matcher() = default;

	Matcher(MType t) {};

	void run(const shared_ptr<Features> f1, const shared_ptr<Features> f2){
		int min_ = min(min(
			(int) f1->m_descriptors->m_descp.rows(), 
			(int) f2->m_descriptors->m_descp.rows()
		), 5);

		//auto argsMax_1 = f1->m_keypoints->

		//auto diff = f1->m_descriptors->m_descp - f2->m_descriptors->m_descp;

		//std::cout << diff.row(0);
		
	}


};

#endif MATCHER_H