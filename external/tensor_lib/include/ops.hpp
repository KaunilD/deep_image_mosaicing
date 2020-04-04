
struct add_op {
	__host__ __device__ static inline float op(const float& a, const float& b) {
		return a + b;
	}
}; 
struct sub_op {
	__host__ __device__ static inline float op(const float& a, const float& b) {
		return a - b;
	}
};
struct div_op {
	__host__ __device__ static inline float op(const float& a, const float& b) {
		return a/b;
	}
};
struct mul_op {
	__host__ __device__ static inline float op(const float& a, const float& b) {
		return a * b;
	}
};