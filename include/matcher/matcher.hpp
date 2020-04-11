#ifndef MATCHER_H
#define MATCHER_H

#include "libs.hpp"
#include "dtypes/features.hpp"
#include "dtypes/matchpair.hpp"

class Matcher {
public:
	Matcher();

	virtual std::vector<MatchPair> run(const std::vector<Descriptor>& d1, const std::vector<Descriptor>& d2) = 0;


};

#endif MATCHER_H