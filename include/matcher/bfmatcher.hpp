#ifndef BFMATCHER_H
#define BFMATCHER_H

#include "libs.hpp"
#include "dtypes/features.hpp"
#include "dtypes/matchpair.hpp"

class BFMatcher{
public:
	BFMatcher(){};

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
				float distance = (d1.at(i).m_descp - d2.at(j).m_descp).squaredNorm();
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

			if (minDistance1 < ratio * minDistance2) {
				matches.push_back({ i, matchSoFar, minDistance1 });
			}
		}
		
		return std::move(matches);
	};
};

#endif BFMATCHER_H