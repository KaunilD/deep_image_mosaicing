#ifndef ESTIMATOR_H
#define ESTIMATOR_H
#include "dtypes/keypoint.hpp"
#include "dtypes/matchpair.hpp"
#include "dtypes/estimatorres.hpp"

class Estimator {
	/*
		http://www.cse.psu.edu/~rtc12/CSE486/lecture20_6pp.pdf
		http://www.cs.cmu.edu/afs/andrew/scs/cs/15-463/f07/proj_final/www/amichals/fundamental.pdf
	*/
public:
	Estimator() = default;
	EstimatorRes normalize(const std::vector<Keypoint>& kps) {
		std::vector<Keypoint> normedKps;

		Eigen::Vector2f centroid(0.0f, 0.0f);
		for (const auto& kp : kps) {
			centroid += kp.m_loc;
		}
		centroid /= kps.size();

		float distance = 0.0f;
		for (size_t i = 0; i < kps.size(); i++) {
			Keypoint kp(kps.at(i).m_loc - centroid, kps.at(i).m_conf);
			normedKps.push_back(std::move(kp));

			distance += normedKps.at(i).m_loc.norm();
		}
		distance /= kps.size();

		float scale = sqrt(2) / distance;

		for (size_t i = 0; i < kps.size(); i++) {
			normedKps.at(i).m_loc *= scale;
		}

		Eigen::Matrix<float, 3, 3> T;
		T << scale, 0.0f, -scale * centroid(0),
			0.0f, scale, -scale * centroid(1),
			0.0f, 0.0f, 1.0f;

		return { std::move(T), std::move(normedKps) };
	}

	void estimateEightPoints(
		const std::vector<Keypoint>& kps1,
		const std::vector<Keypoint>& kps2,
		const std::vector<MatchPair>& matches
	) {
		//assert(kp1.size() == kp2.size());

		auto normedKp1 = normalize(kps1);
		auto normedKp2 = normalize(kps2);

		Eigen::MatrixXf matKps1(2, matches.size()), matKps2(2, matches.size());
		for (int i = 0; i < matches.size(); i++) {

			auto kp1 = normedKp1.kps.at(matches.at(i).kp1);
			auto kp2 = normedKp1.kps.at(matches.at(i).kp2);

			matKps1.col(i) = kp1.m_loc;
			matKps2.col(i) = kp2.m_loc;
		}

		// Af = 0;
		Eigen::MatrixXf A(9, matches.size());
		A << matKps2.row(0).array() * matKps1.row(0).array(),
			 matKps2.row(0).array()* matKps1.row(1).array(),
			 matKps2.row(0).array(),
			 matKps2.row(1).array()* matKps1.row(0).array(),
			 matKps2.row(1).array()* matKps1.row(1).array(),
			 matKps2.row(1).array(),
			 matKps1.row(0),
			 matKps1.row(1),
			 matKps1.row(1),
			 Eigen::MatrixXf::Ones(1, matches.size());

		A = A.transpose().eval();

		Eigen::Matrix<float, 9, 1> fV;

		if (matches.size() == 8) {
			// LU decomp
		}
		else {
			Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, 9> > UDV(A, Eigen::ComputeFullV);
			fV = UDV.matrixV().col(8);
		}

		Eigen::Matrix<float, 3, 3> F;

		F << fV(0), fV(1), fV(2),
			 fV(3), fV(4), fV(5),
			 fV(6), fV(7), fV(8);

		// Singularity constraint
		Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> UDV_S(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		auto sigma = UDV_S.singularValues();
		sigma(2) = 0.0f;
		F = UDV_S.matrixU() * sigma.asDiagonal() * UDV_S.matrixV().transpose();
		F = matKps2.transpose() * F * matKps1;

	};


};


#endif ESTIMATOR_H