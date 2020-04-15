#ifndef BFMATCHER_H
#define BFMATCHER_H

#include "libs.hpp"
#include "dtypes/features.hpp"
#include "dtypes/matchpair.hpp"

class BFMatcher{
public:
	bool m_2_way = false;
	float m_ratio = 0.7f;

	BFMatcher(bool t_2_way) : m_2_way(t_2_way) {

	};

	BFMatcher(float t_ratio) : m_ratio(t_ratio) {

	};

	std::vector<MatchPair> run(const std::vector<Descriptor>& d1, const std::vector<Descriptor>& d2);

	/*
		Returns the index and distance of the closest descriptor in d2 
	*/
	std::tuple<int, float> match(const Descriptor& d1, const std::vector<Descriptor>& d2);
};

#endif BFMATCHER_H