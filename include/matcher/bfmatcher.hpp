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

	std::vector<MatchPair> run(const std::vector<Descriptor>& d1, const std::vector<Descriptor>& d2){

		float minDistance1 = 1e8; // first nearest
		float minDistance2 = 1e8; // second nearest 

		float ratio = 0.9f;

		std::vector<MatchPair> matches;

		for (int i = 0; i < d1.size(); i++) {
			int matchSoFar = 0;
			minDistance1 = minDistance2 = 1e8;
			for (int j = 0; j < d2.size(); j++) {
				//std::cout <<  d1.at(i).m_descp.size() << " " << d2.at(j).m_descp.size() << "\n";
				float distance = (d1.at(i).m_descp - d2.at(j).m_descp).lpNorm<1>();
				//std::cout << distance << "\n";
				if (distance < minDistance1) {
					minDistance2 = minDistance1;
					minDistance1 = distance;
					matchSoFar = j;
				}
				else if (distance < minDistance2) {
					minDistance2 = distance;
				}
			}
			if (!m_2_way) {

				auto res = match(d2.at(matchSoFar), d1);
				if (std::get<0>(res) == i) {
					matches.push_back({ i, matchSoFar, minDistance1 });
				}

			}
			else {
				if (minDistance1 < ratio * minDistance2) {
					matches.push_back({ i, matchSoFar, minDistance1 });
				}
			}
		}
		
		return std::move(matches);
	};

	/*
		Returns the index and distance of the closest descriptor in d2 
	*/
	std::tuple<int, float> match(const Descriptor& d1, const std::vector<Descriptor>& d2) {
		float minDistance = 1e8;
		int index = 0;

		for (int i = 0; i < d2.size(); i++) {
			auto distance = (d1.m_descp - d2.at(i).m_descp).lpNorm<1>();
			if (distance < minDistance) {
				minDistance = distance;
				index = i;
			}
		}
		auto res = std::make_tuple(index, minDistance);
		return std::move(res);

	}
};

#endif BFMATCHER_H