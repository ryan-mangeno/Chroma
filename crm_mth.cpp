#include "crm_mth.h"
#include <cmath>


// used material from getintogamedev on youtube
// https://www.youtube.com/@GetIntoGameDev/videos

// might end up splitting up into different src files based on functionality
// vector ops, transformations, mat stuff, etc


/* to add:
* 

	vec2 operations, dot, add, mul, sub, etc

	rotating view matrix, with some rotation, about some vector:

	-> rotate(View, Rotate.y, vec3(-1.0f, 0.0f, 0.0f));



*/


namespace crm {


	/*--------Constructors---------------*/

	vec2::vec2(float x, float y)
		: x(x), y(y)
	{
	}

	bvec2::bvec2(float x, float y)
		: x(x), y(y), _pad0(0.0f), _pad1(0.0f)
	{
	}

	vec3::vec3(float x, float y, float z)
		: x(x), y(y), z(z), _pad0(0.0f)
	{
	}

	vec4::vec4(float x, float y, float z, float w)
		: x(x), y(y), z(z), w(w)
	{
	}


	mat4::mat4(float x, float y, float z, float w) {
		/*
		
		x 0 0 0
		0 y 0 0
		0 0 z 0
		0 0 0 w

		*/
		// didnt want to use simd for unnecesary loads, simd will do the same thing
		column_vector[0] = vec4(x, 0.0f, 0.0f, 0.0f);
		column_vector[1] = vec4(0.0f, y, 0.0f, 0.0f);
		column_vector[2] = vec4(0.0f, 0.0f, z, 0.0f);
		column_vector[3] = vec4(0.0f, 0.0f, 0.0f, w);
	}


	quat::quat()
		: x(0.0f), y(0.0f), z(0.0f), w(1.0f)
	{
	}

	quat::quat(float x, float y, float z, float w)
		: x(x), y(y), z(z), w(w)
	{
	}

	// making quaternion from rotation
	quat::quat(float angle, const vec3& axis) {

		vec3 normAxis = Normalize(axis);
		float s = sinf(radians(angle / 2));
		float c = cosf(radians(angle / 2));

		vector = _mm_mul_ps(normAxis.vector, _mm_set1_ps(s));
		data[3] = c;

	}

	// making rotation from two vec3's
	quat::quat(const vec3& a, const vec3& b) : w(0.0f) {
		vec3 aNorm = Normalize(a);
		vec3 bNorm = Normalize(b);

		// a and b might be antiparallel
		if (Close(aNorm, Mul(bNorm, -1.0f))) {

			//we want to  around a to get to b,
			//pick the least dominant component as the rotation direction
			vec3 ortho(1, 0, 0);
			if (fabsf(aNorm.data[1]) < fabs(aNorm.data[0])) {
				ortho = vec3(0, 1, 0);
			}
			if (fabsf(aNorm.data[2]) < std::fmin(fabs(aNorm.data[0]), fabs(aNorm.data[1]))) {
				ortho = vec3(0, 0, 1);
			}
			vec3 axis = Normalize(Cross(aNorm, ortho));

			x = axis.data[0];
			y = axis.data[1];
			z = axis.data[2];
		}

		else
		{
			//Construct the regular quaternion
			vec3 halfVec = Normalize(Add(aNorm, bNorm));
			vec3 axis = Cross(aNorm, halfVec);

			x = axis.data[0];
			y = axis.data[1];
			z = axis.data[2];
			w = Dot(aNorm, halfVec);

		}
	}


	/*--------Utility Operations---------*/


	/*
		The fast inv sqrt by Terje Mathisen and Gary Tarolli ...
		"Use (almost) the same algorithm as CPU/FPU,
		exploit the improvement of initial conditions for the special case of 1/sqrt(x)
		and don't calculate all the way to precision CPU/FPU will go to but stop earlier,
		thus gaining in calculation speed." - BJovke (stack overflow)

		https://stackoverflow.com/questions/1349542/john-carmacks-unusual-fast-inverse-square-root-quake-iii
	*/
	float fast_inv_sqrt(float x) {
		float half = 0.5f * x;

		int i = *(int*)&x; // Treat float as integer to use bit-level manipulation

		i = 0x5f3759df - (i >> 1); // Magic number: 0x5f3759df is a constant for the approximation

		x = *(float*)&i; // Convert back to float

		x = x * (1.5f - half * x * x); // One iteration of Newton’s method for refinement

		return x;
	}

	float fast_sqrt(float x) {
		return 1.0f / fast_inv_sqrt(x);
	}

	/*
		info on why to avoid calling floor()
		https://stackoverflow.com/questions/2352303/avoiding-calls-to-floor


		_mm_load_ss(&f) takes the input float f and places it in an SSE register.
		_mm_cvtt_ss2si converts the value in the SSE register to an integer using truncation.
		The resulting integer is returned.
	*/
	int fast_ftoi(float f)
	{
		return _mm_cvtt_ss2si(_mm_load_ss(&f));
	}



	/*-------- Conversions----------*/

	float radians(float angle) {
		return angle * pi / 180.0f;
	}

	float degrees(float angle) {
		return angle * 180.0f / pi;
	}

	/*-------- Vec3 Operations----------*/


	float AngleBetweenVectors3(const vec3& a, const vec3& b) {

		/*
		consider two vectors a and b

		the |projection| of a onto b is, cos(theta)|a|
		scaling the projection by |b| gives us the dot product being
		dot(a,b) = |a||b|cos(theta)
		solving for theta, we get dot(a,b)/(|a||b|)


		*/

		float denominator = sqrtf(Dot(a, a) * Dot(b, b));

		float dot = Dot(a, b);

		return degrees(acosf(dot / denominator));
	}


	float Dot(const vec3& a, const vec3& b) {
		return a.data[0] * b.data[0] + a.data[1] * b.data[1] + a.data[2] * b.data[2];
	}

	vec3 Cross(const vec3& a, const vec3& b) {
		vec3 result;

		// cross product gives the perpendicular vector to two vectors 

		result.data[0] = a.data[1] * b.data[2] - a.data[2] * b.data[1];
		result.data[1] = a.data[2] * b.data[0] - a.data[0] * b.data[2];
		result.data[2] = a.data[0] * b.data[1] - a.data[1] * b.data[0];
		result.data[3] = 0;

		return result;
	}

	vec3 Normalize(const vec3& a) {

		// get inv magnitude, we will take this scalar and mult to vector, mag will be 1

		float invMagnitude = 1.0f / sqrtf(a.data[0] * a.data[0] + a.data[1] * a.data[1] + a.data[2] * a.data[2]);

		vec3 result;

		// _mm_mul_ps used to mult two 128 bit simd registers
		// set1_ps sets a the 4 floats in 128 bit register to said input float
		result.vector = _mm_mul_ps(a.vector, _mm_set1_ps(invMagnitude));

		return result;
	}

	vec3 Sub(const vec3& a, const vec3& b) {

		vec3 result;

		result.vector = _mm_sub_ps(a.vector, b.vector);

		return result;
	}

	vec3 Add(const vec3& a, const vec3& b) {

		vec3 result;

		result.vector = _mm_add_ps(a.vector, b.vector);

		return result;
	}

	vec3 Mul(const vec3& a, float scalar) {

		vec3 result;

		result.vector = _mm_mul_ps(a.vector, _mm_set1_ps(scalar));

		return result;

	}


	vec3 Project(const vec3& incoming, const vec3& basis) {
		/*
		 take dot of incoming and basis, giving us |a||b|cos(theta)
		 divide by |b| to get unscaled magnitude of the projection
		 |projection| = |a|cos(theta)
		 projection = |a|cos(theta) * b/|b| to get projection in direction of b
		 we can simply take dot(b,b) to get |b|^2, we will divide |a||b|cos(theta) by |b|^2 then mul by b
		 this gets our projection
		*/
		return Mul(basis, Dot(incoming, basis) / Dot(basis, basis));
	}

	vec3 Reject(const vec3& incoming, const vec3& basis) {
		return Sub(incoming, Project(incoming, basis));
	}

	vec3 Reflect(const vec3& incident, const vec3& normal) {

		vec3 result;
		// reflected = incident − 2(incident.normal)normal

		//_mm_fmadd_ps takes 3 parameters and performs a fused multiply-add operation on packed floats
		// multiplies src1 and src2 , adds src3

		/*


		we get projection of inci and norm with dot(inci, norm)/|norm|^2 * norm
		we have some vec c, that is perpendicular to the projection vec ( rejection vec )
		we know -c too, which will be useful for the reflection vec
		proj + (-c) gives the reflection vec

		for the full formula, we have

		dot(inci, norm)/|norm|^2 * norm + dot(inci, norm)/|norm|^2 * norm - inci
		or
		2(dot(inci, norm)/|norm|^2 * norm) - a


		however, since we are dealing with physics, with rays, we  replace inci with -inci, simply flipping the sign
		since the ray is not going out of a wall for example but into it
		flipping the signs

		-2(dot(inci, norm)/|norm|^2 * norm) + a

		*/
		result.vector = _mm_fmadd_ps(
			normal.vector,
			_mm_set1_ps(-2.0f * Dot(incident, normal)),
			incident.vector
		);

		return result;
	}

	vec3 Lerp(const vec3& a, const vec3& b, float t) {

		vec3 result;

		// linear interpolation between two vecs
		/*

			when t = 0, its a
			t = 0, its b

			lerp(a,b,t) = (1-t)(a) + (t)(b)
			expanding
			a - at + bt

			a + t(b-a)
			t(b-a) + a

			so we subtract b-a and multiply by t, then add a
			allowed to us by _mm_fmadd_ps to multiply to things then add the third
		*/

		result.vector = _mm_fmadd_ps(
			_mm_sub_ps(b.vector, a.vector),
			_mm_set1_ps(t),
			a.vector
		);

		return result;
	}

	vec3 Slerp(const vec3& a, const vec3& b, float t) {

		// spherical linear interp
		// special case when t<.1, just use linear interp
		// SLERP(a, b, t) = (sin((1-t) * theta) / sin(theta)) * a + (sin(t * theta) / sin(theta)) * b

		if (t < 0.1f) {
			return Lerp(a, b, t);
		}

		float angle = AngleBetweenVectors3(a, b);

		float denominator = sinf(radians(angle));

		vec3 result;

		result.vector = _mm_fmadd_ps(
			a.vector,
			_mm_set1_ps(sinf(1 - t) * angle / denominator),
			_mm_mul_ps(b.vector, _mm_set1_ps(sinf(t * angle) / denominator))
		);

		return result;
	}

	// normalizing lerp vec
	vec3 Nlerp(const vec3& a, const vec3& b, float t) {
		return Normalize(Lerp(a, b, t));
	}


	// ease in for lerp animation
	// slow then fast
	float ease_in(float t) {
		return t * t;
	}

	// ease out for lerp animation
	// fast then slow
	float ease_out(float t) {
		return t * (2 - t);
	}


	bool Close(const vec3& a, const vec3& b) {

		// get the displacement vec to see how far the terminal
		// points vary from the other, dot product with itself to get the mag
		vec3 displacement = Sub(a, b);

		return Dot(displacement, displacement) < eps;
	}

	/*-------- Vector4 Operations ----------*/

	float Dot(const vec4& a, const vec4& b) {
		return a.data[0] * b.data[0] + a.data[1] * b.data[1] + a.data[2] * b.data[2] + a.data[3] * b.data[3];
	}

	vec4 Normalize(const vec4& a) {

		float invMagnitude = 1.0f / sqrtf(
			a.data[0] * a.data[0] + a.data[1] * a.data[1] + a.data[2] * a.data[2] + a.data[3] * a.data[3]
		);

		vec4 result;

		result.vector = _mm_mul_ps(a.vector, _mm_set1_ps(invMagnitude));

		return result;
	}

	/*-------- Matrix4 Operations ----------*/


	mat4 MakePerspectiveProjection( float fovy, float aspect, float near, float far) {

		float yMax = near * tanf(radians(fovy / 2));
		float xMax = yMax * aspect;

		/*

			The matrix is:

			[E 0  A 0]
			[0 F  B 0]
			[0 0  C D]
			[0 0 -1 0]

			Given by:

			float left{ -xMax };
			float right{ -xMax };
			float top{ -yMax };
			float bottom{ yMax };

			float A{ (right + left) / (right - left) };
			float B{ (top + bottom) / (top - bottom) };
			float C{ -(far + near) / (far - near) };
			float D{ -2.0f * far * near / (far - near) };
			float E{ 2.0f * near / (right - left) };
			float F{ 2.0f * near / (top - bottom) };

			(In practice this simplifies out quite a bit though.)
		*/
		float C = -(far + near) / (far - near);
		float D = -2.0f * far * near / (far - near);
		float E = near / xMax;
		float F = near / yMax;

		mat4 result;

		result.column[0] = _mm_setr_ps(E, 0, 0, 0);
		result.column[1] = _mm_setr_ps(0, F, 0, 0);
		result.column[2] = _mm_setr_ps(0, 0, C, -1);
		result.column[3] = _mm_setr_ps(0, 0, D, 0);

		return result;
	}

	mat4 LookAt(const vec3& eye, const vec3& target, const vec3& up) {

		vec3 forwardsNorm = Normalize(Sub(target, eye));
		vec3 rightNorm = Normalize(Cross(forwardsNorm, up));
		vec3 upNorm = Normalize(Cross(rightNorm, forwardsNorm));
		vec3 backwardNorm = Mul(forwardsNorm, -1);

		mat4 result;

		result.column[0] = _mm_setr_ps(rightNorm.data[0], upNorm.data[0], backwardNorm.data[0], 0);
		result.column[1] = _mm_setr_ps(rightNorm.data[1], upNorm.data[1], backwardNorm.data[1], 0);
		result.column[2] = _mm_setr_ps(rightNorm.data[2], upNorm.data[2], backwardNorm.data[2], 0);
		result.column[3] = _mm_setr_ps(-Dot(rightNorm, eye), -Dot(upNorm, eye), -Dot(backwardNorm, eye), 1);

		return result;
	}


	mat4 Translation(const vec3& translation) {

		mat4 result;

		// w = 1 for coordinates, w = 0 for directions
		// first three columns default constructed for identity matrix
	  //result.column[0] = _mm_setr_ps(1, 0, 0, 0);
	  //result.column[1] = _mm_setr_ps(0, 1, 0, 0);
	  //result.column[2] = _mm_setr_ps(0, 0, 1, 0);
		result.column[3] = _mm_setr_ps(translation.data[0], translation.data[1], translation.data[2], 1);

		return result;
	}

	mat4 Translation(const mat4& m, const vec3& translation) {

		mat4 result = m;
		
		// just setting the 4th column and adding possible existing translations

		// for vec3, layout will be <x,y,z,0>, as long as the user doesnt touch the pad
		
		/*
		<x, y, z, 0>    +<tx, ty, tz, w>
			... <tx + x, ty + y, tz + z, 0 + w>

		again, assuming the 0 in the vec3, if the user doesnt touch the padding
		*/
		result.column[3] = _mm_add_ps(translation.vector, result.column[3]);


		/*
		alternative, safer solution, but some overhead

		result.column[3] = _mm_add_ps(_mm_setr_ps(translation.data[0], translation.data[1], translation.data[2], 1),
								(result.column[3]));

		*/

		return result;
	}

	mat4 XRotation(float angle) {

		angle = radians(angle);
		float cT = cosf(angle);
		float sT = sinf(angle);
		
		// dont know if i want this, with cosf there is precision errors with pi/2 or 90 deg
		// cT = (cT < eps && cT > -eps) ? 0 : cT;

		mat4 result(1.0f);


		// first and last column are default constructed
	//  result.column[0] = _mm_setr_ps(1,  0,   0, 0);


		result.column_vector[1].x = 0.0f;
		result.column_vector[1].y = cT;
		result.column_vector[1].z = -sT;
		result.column_vector[1].w = 0.0f;

		result.column_vector[2].x = 0.0f;
		result.column_vector[2].y = sT;
		result.column_vector[2].z = cT;
		result.column_vector[2].w = 0.0f;

	//	result.column[3] = _mm_setr_ps(0,  0,   0, 1);


		return result;
	}

	mat4 YRotation(float angle) {

		angle = radians(angle);
		float cT = cosf(angle);
		float sT = sinf(angle);

		mat4 result;

		result.column[0] = _mm_setr_ps(cT, 0, sT, 0);
	//  result.column[1] = _mm_setr_ps( 0,  1, 0, 0);
		result.column[2] = _mm_setr_ps(-sT, 0, cT, 0);
	//  result.column[3] = _mm_setr_ps( 0,  0, 0, 1);


		return result;
	}

	mat4 ZRotation(float angle) {

		angle = radians(angle);
		float cT = cosf(angle);
		float sT = sinf(angle);

		mat4 result;

		result.column[0] = _mm_setr_ps(cT, -sT, 0, 0);
		result.column[1] = _mm_setr_ps(sT, cT, 0, 0);
		//	result.column[2] = _mm_setr_ps( 0,   0, 1, 0);
		//	result.column[3] = _mm_setr_ps( 0,   0, 0, 1);

		return result;
	}

	vec4 Mul(const mat4& m, const vec4& v) {

		vec4 result;

		/*
		v.data[0] is multiplied by each element in m.column[0]
		v.data[1] is multiplied by each element in m.column[1]
		v.data[2] is multiplied by each element in m.column[2]
		v.data[3] is multiplied by each element in m.column[3]

		add all vectors resulting vectors from simd ...
		*/

		result.vector = _mm_fmadd_ps(_mm_set1_ps(v.data[0]), m.column[0],
			_mm_fmadd_ps(_mm_set1_ps(v.data[1]), m.column[1],
				_mm_fmadd_ps(_mm_set1_ps(v.data[2]), m.column[2],
					_mm_mul_ps(_mm_set1_ps(v.data[3]), m.column[3])
				)
			)
		);

		return result;
	}


	mat4 Mul(const mat4& m2, const mat4& m1) {

		mat4 result;

		// this is not a recursive call ...
		// we call it on the mat column vector, so we multiply the column vector by
		// all 4 column vectors in m2
		result.column_vector[0] = Mul(m2, m1.column_vector[0]);
		result.column_vector[1] = Mul(m2, m1.column_vector[1]);
		result.column_vector[2] = Mul(m2, m1.column_vector[2]);
		result.column_vector[3] = Mul(m2, m1.column_vector[3]);

		return result;
	}

	mat4 Add(const mat4& m1, const mat4& m2) {

		mat4 m3;
		m3.chunk[0] = _mm256_add_ps(m1.chunk[0], m2.chunk[0]);
		m3.chunk[1] = _mm256_add_ps(m1.chunk[1], m2.chunk[1]);

		return m3;
	}

	mat4 Mul(const mat4& matrix, float scalar) {

		mat4 m3;
		__m256 scale = _mm256_set1_ps(scalar);
		m3.chunk[0] = _mm256_mul_ps(matrix.chunk[0], scale);
		m3.chunk[1] = _mm256_mul_ps(matrix.chunk[1], scale);

		return m3;
	}

	mat4 Lerp(const mat4& m1, const mat4& m2, float t) {

		mat4 m3;
		__m256 scale = _mm256_set1_ps(t);

		m3.chunk[0] = _mm256_fmadd_ps(
			_mm256_sub_ps(m2.chunk[0], m1.chunk[0]),
			scale,
			m1.chunk[0]
		);

		m3.chunk[1] = _mm256_fmadd_ps(
			_mm256_sub_ps(m2.chunk[1], m1.chunk[1]),
			scale,
			m1.chunk[1]
		);

		return m3;
	}

	mat4 Transpose(const mat4& matrix) {

		mat4 transposed;

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				transposed.data[i + 4 * j] = matrix.data[j + 4 * i];
			}
		}

		return transposed;
	}

	mat4 TransformInverse(const mat4& matrix) {

		//Get the scale factors
		float a = sqrtf(Dot(matrix.column_vector[0], matrix.column_vector[0]));
		float b = sqrtf(Dot(matrix.column_vector[1], matrix.column_vector[1]));
		float c = sqrtf(Dot(matrix.column_vector[2], matrix.column_vector[2]));

		//Get the rotation vectors, apply inverse scaling
		vec4 X = Normalize(matrix.column_vector[0]);
		X.vector = _mm_mul_ps(X.vector, _mm_set1_ps(1 / a));
		vec4 Y = Normalize(matrix.column_vector[1]);
		Y.vector = _mm_mul_ps(Y.vector, _mm_set1_ps(1 / b));
		vec4 Z = Normalize(matrix.column_vector[2]);
		Z.vector = _mm_mul_ps(Z.vector, _mm_set1_ps(1 / c));
		vec4 T = Normalize(matrix.column_vector[3]);
		T.vector = _mm_mul_ps(T.vector, _mm_set1_ps(-1));

		mat4 inverse;

		//Column adjustments
		inverse.column[0] = _mm_setr_ps(X.data[0], Y.data[0], Z.data[0], 0);
		inverse.column[1] = _mm_setr_ps(X.data[1], Y.data[1], Z.data[1], 0);
		inverse.column[2] = _mm_setr_ps(X.data[2], Y.data[2], Z.data[2], 0);
		inverse.column[3] = _mm_setr_ps(Dot(X, T), Dot(Y, T), Dot(Z, T), 1);

		return inverse;
	}

	/*-------- Quaternion Operations ----------*/

	/*
	old, made constructor for quaternion for x,y,z,and w components
	*/
	// quat MakeQuaternionFromComponents(float x, float y, float z, float w) {
	// 	quat q;
	// 
	// 	q.vector = _mm_setr_ps(x, y, z, w);
	// 
	// 	return q;
	// }



	vec3 GetAxisFromQuaternion(const quat& q) {
		return Normalize(vec3(q.data[0], q.data[1], q.data[2]));
	}


	float GetAngleFromQuaternion(const quat& q) {
		// not scalar first order for quaternion <x,y,z,w>
		return degrees(2.0f * acosf(q.data[3]));
	}

	quat Add(const quat& q1, const quat& q2) {
		quat result;

		result.vector = _mm_add_ps(q1.vector, q2.vector);

		return result;
	}


	quat Sub(const quat& q1, const quat& q2) {
		quat result;

		result.vector = _mm_sub_ps(q1.vector, q2.vector);

		return result;
	}


	quat Mul(const quat& q, float scalar) {
		quat result;

		result.vector = _mm_mul_ps(q.vector, _mm_set1_ps(scalar));

		return result;
	}


	float Dot(const quat& q1, const quat& q2) {
		return q1.data[0] * q2.data[0]
			+ q1.data[1] * q2.data[1]
			+ q1.data[2] * q2.data[2]
			+ q1.data[3] * q2.data[3];
	}


	bool Close(const quat& q1, const quat& q2) {

		quat displacement = Sub(q2, q1);

		return Dot(displacement, displacement) < eps;
	}


	bool QuatSameOrientation(const quat& q1, const quat& q2) {

		quat displacement = Sub(q1, q2);

		if (Dot(displacement, displacement) < eps) {
			return true;
		}

		displacement = Add(q1, q2);

		if (Dot(displacement, displacement) < eps) {
			return true;
		}

		return false;
	}


	quat Normalize(const quat& q) {

		float scalar = 1 / sqrtf(Dot(q, q));

		return Mul(q, scalar);
	}


	quat GetConjQuat(const quat& q) {

		return quat(
			-q.data[0],
			-q.data[1],
			-q.data[2],
			 q.data[3]
		);
	}


	quat InvQuat(const quat& q) {
		return Mul(GetConjQuat(q), 1 / Dot(q, q));
	}


}