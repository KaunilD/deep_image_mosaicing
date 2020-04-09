
class FeatureExtractor {
public:
	FeatureExtractor() = default;

	enum EType {
		ORB,
		SIFT,
		SuperPoint
	};

	virtual void init() = 0;
};