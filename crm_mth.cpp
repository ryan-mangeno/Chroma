#include "crm_mth.h"
#include <cmath>
#include <iostream>

// used material from getintogamedev on youtube
// https://www.youtube.com/@GetIntoGameDev/videos

// might end up splitting up into different src files based on functionality
// vector ops, transformations, mat stuff, etc


/* to add:

	
	clamp

	some specific function i want to add -> rotating arbitary mat4 about some arbitrary bvec3 axis
	-> rotate(mat4, Rotate.y, bvec3(x,y,z));




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
		: x(x), y(y), z(z)
	{
	}


	bvec3::bvec3(float x, float y, float z)
		: x(x), y(y), z(z), _pad0(0.0f)
	{
	}

	bvec3::bvec3(const vec3& v)
		: x(v.x), y(v.y), z(v.z), _pad0(0.0f)
	{
	}

	vec4::vec4(float x, float y, float z, float w)
		: x(x), y(y), z(z), w(w)
	{
	}


	mat3::mat3(float x, float y, float z)
	{
		// we still have 4 elements per column, but the 4th element in each col is pad
		column_vector[0] = vec3(x, 0.0f, 0.0f);
		column_vector[1] = vec3(0.0f, y, 0.0f);
		column_vector[2] = vec3(0.0f, 0.0f, z);
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

		// upscaling for simd operations 
		bvec3 optAxis(axis);

		bvec3 normAxis = Normalize(optAxis);
		float s = sinf(radians(angle / 2));
		float c = cosf(radians(angle / 2));

		vector = _mm_mul_ps(optAxis.vector, _mm_set1_ps(s));
		data[3] = c;

	}

	quat::quat(const vec3& a, const vec3& b) : w(0.0f) {
		bvec3 aNorm = Normalize(a);
		bvec3 bNorm = Normalize(b);

		// a and b might be antiparallel
		if (Close(aNorm, Mul(bNorm, -1.0f))) {

			//we want to  around a to get to b,
			//pick the least dominant component as the rotation direction
			bvec3 ortho(1, 0, 0);
			if (fabsf(aNorm.data[1]) < fabs(aNorm.data[0])) {
				ortho = bvec3(0, 1, 0);
			}
			if (fabsf(aNorm.data[2]) < std::fmin(fabs(aNorm.data[0]), fabs(aNorm.data[1]))) {
				ortho = bvec3(0, 0, 1);
			}
			bvec3 axis = Normalize(Cross(aNorm, ortho));

			x = axis.data[0];
			y = axis.data[1];
			z = axis.data[2];
		}

		else
		{
			//Construct the regular quaternion
			bvec3 halfVec = Normalize(Add(aNorm, bNorm));
			bvec3 axis = Cross(aNorm, halfVec);

			x = axis.data[0];
			y = axis.data[1];
			z = axis.data[2];
			w = Dot(aNorm, halfVec);

		}
	}

	// making quaternion from rotation
	quat::quat(float angle, const bvec3& axis) {

		bvec3 normAxis = Normalize(axis);
		float s = sinf(radians(angle / 2));
		float c = cosf(radians(angle / 2));

		vector = _mm_mul_ps(normAxis.vector, _mm_set1_ps(s));
		data[3] = c;

	}

	// making rotation from two bvec3's
	quat::quat(const bvec3& a, const bvec3& b) : w(0.0f) {
		bvec3 aNorm = Normalize(a);
		bvec3 bNorm = Normalize(b);

		// a and b might be antiparallel
		if (Close(aNorm, Mul(bNorm, -1.0f))) {

			//we want to  around a to get to b,
			//pick the least dominant component as the rotation direction
			bvec3 ortho(1, 0, 0);
			if (fabsf(aNorm.data[1]) < fabs(aNorm.data[0])) {
				ortho = bvec3(0, 1, 0);
			}
			if (fabsf(aNorm.data[2]) < std::fmin(fabs(aNorm.data[0]), fabs(aNorm.data[1]))) {
				ortho = bvec3(0, 0, 1);
			}
			bvec3 axis = Normalize(Cross(aNorm, ortho));

			x = axis.data[0];
			y = axis.data[1];
			z = axis.data[2];
		}

		else
		{
			//Construct the regular quaternion
			bvec3 halfVec = Normalize(Add(aNorm, bNorm));
			bvec3 axis = Cross(aNorm, halfVec);

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

	/*-------- Vec2 Operations-----------*/

	float AngleBetweenVectors2(const vec2& a, const vec2& b) {

		/*
		consider two vectors a and b

		the |projection| of a onto b is, cos(theta)|a|
		scaling the projection by |b| gives us the dot product being
		dot(a,b) = |a||b|cos(theta)
		solving for theta, we get dot(a,b)/(|a||b|)


		*/

		float denominator = sqrtf(Dot(a, a) * Dot(b, b));

		// case when the magnitude of vector a or vector b is 0 is undefined
		if (denominator == 0.0f) {
			return 0.0f;
		}

		float dot = Dot(a, b);

		return degrees(acosf(dot / denominator));
	}

	float AngleBetweenVectors2(const bvec2& a, const bvec2& b) {

		float denominator = sqrtf(Dot(a, a) * Dot(b, b));

		// case when the magnitude of vector a or vector b is 0 is undefined
		if (denominator == 0.0f) {
			return 0.0f;
		}

		float dot = Dot(a, b);

		return degrees(acosf(dot / denominator));
	}

	float Dot(const vec2& a, const vec2& b) {
		return a.data[0] * b.data[0] + a.data[1] * b.data[1];
	}

	float Dot(const bvec2& a, const bvec2& b) {
		return a.data[0] * b.data[0] + a.data[1] * b.data[1];
	}

	float Lerp(float a, float b, float t)
	{

		// linear interpolation between two floats
		/*

			when t = 0, its a
			t = 0, its b

			lerp(a,b,t) = (1-t)(a) + (t)(b)
			expanding
			a - at + bt

			a + t(b-a)

			result = t(b-a) + a

		*/


		return (1.0f - t) * (a) + (t) * (b);
	
	}

	vec2 Normalize(const vec2& a) {

		// get inv magnitude, we will take this scalar and mult to vector, mag will be 1

		float invMagnitude = 1.0f / sqrtf(a.data[0] * a.data[0] + a.data[1] * a.data[1]);

		vec2 result;

		// _mm_mul_ps used to mult two 128 bit simd registers
		// set1_ps sets a the 4 floats in 128 bit register to said input float
		result.x = result.x / invMagnitude;
		result.y = result.y / invMagnitude;

		return result;
	}

	bvec2 Normalize(const bvec2& a) {

		// get inv magnitude, we will take this scalar and mult to vector, mag will be 1

		float invMagnitude = 1.0f / sqrtf(a.data[0] * a.data[0] + a.data[1] * a.data[1]);

		bvec2 result;

		// _mm_mul_ps used to mult two 128 bit simd registers
		// set1_ps sets a the 4 floats in 128 bit register to said input float
		result.vector = _mm_mul_ps(a.vector, _mm_set1_ps(invMagnitude));

		return result;
	}

	vec2 Sub(const vec2& a, const vec2& b) {

		vec2 result;

		result.x = (a.x - b.x);
		result.y = (a.y - b.y);

		return result;
	}

	bvec2 Sub(const bvec2& a, const bvec2& b) {

		bvec2 result;

		result.vector = _mm_sub_ps(a.vector, b.vector);

		return result;
	}

	vec2 Add(const vec2& a, const vec2& b) {
		return vec2{ a.x + b.x, a.y + b.y };
	}

	bvec2 Add(const bvec2& a, const bvec2& b) {

		bvec2 result;

		result.vector = _mm_add_ps(a.vector, b.vector);

		return result;
	}

	vec2 Mul(const vec2& a, const vec2& b) {
		return { a.x * b.x, a.y * b.y };
	}

	bvec2 Mul(const bvec2& a, const bvec2& b) {
		bvec2 result;

		result.vector = _mm_mul_ps(a.vector, b.vector);

		return result;
	}

	vec2 Mul(const vec2& a, float scalar) {
		return { a.x * scalar, a.y * scalar };
	}

	bvec2 Mul(const bvec2& a, float scalar) {

		bvec2 result;

		result.vector = _mm_mul_ps(a.vector, _mm_set1_ps(scalar));

		return result;

	}


	vec2 Project(const vec2& incoming, const vec2& basis) {
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

	bvec2 Project(const bvec2& incoming, const bvec2& basis) {
		return Mul(basis, Dot(incoming, basis) / Dot(basis, basis));
	}

	vec2 Reject(const vec2& incoming, const vec2& basis) {
		return Sub(incoming, Project(incoming, basis));
	}

	bvec2 Reject(const bvec2& incoming, const bvec2& basis) {
		return Sub(incoming, Project(incoming, basis));
	}

	vec2 Reflect(const vec2& incident, const vec2& normal) {
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



		float dotProduct = incident.x * normal.x + incident.y * normal.y;

		float scale = -2.0f * dotProduct;

		vec2 scaledNormal{ normal.x * scale, normal.y * scale };

		vec2 result {scaledNormal.x + incident.x, scaledNormal.y + incident.y};

		return result;
	}

	bvec2 Reflect(const bvec2& incident, const bvec2& normal) {


		bvec2 result;

		// applying the same formula above, we can optimize heavily with this simd operation
		result.vector = _mm_fmadd_ps(
			normal.vector,
			_mm_set1_ps(-2.0f * Dot(incident, normal)),
			incident.vector
		);

		return result;
	}

	vec2 Lerp(const vec2& a, const vec2& b, float t) {

		vec2 result;

		// linear interpolation between two vecs
		/*

			when t = 0, its a
			t = 0, its b

			lerp(a,b,t) = (1-t)(a) + (t)(b)
			expanding
			a - at + bt

			a + t(b-a)
			
			result = t(b-a) + a

			so we subtract b-a and multiply by t, then add a
			allowed to us by _mm_fmadd_ps to multiply to things then add the third
		*/

		// Compute the element-wise subtraction (b - a)
		vec2 diff(b.x - a.x, b.y - a.y);

		// Scale the difference by t
		vec2 scaledDiff(diff.x * t, diff.y * t);


		// Add the scaled difference to a
		result.x = scaledDiff.x + a.x;
		result.y = scaledDiff.y + a.y;


		return result;
	}

	bvec2 Lerp(const bvec2& a, const bvec2& b, float t) {

		bvec2 result;

		// result = t(b-a) + a
		result.vector = _mm_fmadd_ps(
			_mm_sub_ps(b.vector, a.vector),
			_mm_set1_ps(t),
			a.vector
		);

		return result;
	}

	vec2 Slerp(const vec2& a, const vec2& b, float t) {


		// SLERP(a, b, t) = (sin((1-t) * theta) / sin(theta)) * a + (sin(t * theta) / sin(theta)) * b

		if (t < 0.1f) {
			return Lerp(a, b, t);
		}

		// Compute the angle between the vectors
		float angle = AngleBetweenVectors2(a, b);

		// when angle between two vectors is 0 linear interp is undefined
		if (angle == 0.0f) {
			return { 0.0f, 0.0f };
		}

		float denominator = sinf(radians(angle));

		// Compute the scaling factors
		float scaleA = sinf((1 - t) * angle) / denominator;
		float scaleB = sinf(t * angle) / denominator;

		// Scale vectors
		vec2 scaledA = { a.x * scaleA, a.y * scaleA };
		vec2 scaledB = { b.x * scaleB, b.y * scaleB };

		// Add scaled vectors
		vec2 result = { scaledA.x + scaledB.x, scaledA.y + scaledB.y };

		return result;
	}

	bvec2 Slerp(const bvec2& a, const bvec2& b, float t) {


		// SLERP(a, b, t) = (sin((1-t) * theta) / sin(theta)) * a + (sin(t * theta) / sin(theta)) * b

		if (t < 0.1f) {
			return Lerp(a, b, t);
		}

		float angle = AngleBetweenVectors2(a, b);

		// when angle between two vectors is 0, slerp is undefined, sin(theta) is divison by 0
		if (angle == 0.0f) {
			return { 0.0f, 0.0f };
		}

		float denominator = sinf(radians(angle));


		bvec2 result;

		result.vector = _mm_fmadd_ps(
			a.vector,
			_mm_set1_ps(sinf(1 - t) * angle / denominator),
			_mm_mul_ps(b.vector, _mm_set1_ps(sinf(t * angle) / denominator))
		);

		return result;
	}

	// normalizing lerp vec
	vec2 Nlerp(const vec2& a, const vec2& b, float t) {
		return Normalize(Lerp(a, b, t));
	}


	bvec2 Nlerp(const bvec2& a, const bvec2& b, float t) {
		return Normalize(Lerp(a, b, t));
	}


	bool Close(const vec2& a, const vec2& b) {

		// get the displacement vec to see how far the terminal
		// points vary from the other, dot product with itself to get the mag
		vec2 displacement = Sub(a, b);

		return Dot(displacement, displacement) < eps;
	}

	bool Close(const bvec2& a, const bvec2& b) {

		bvec2 displacement = Sub(a, b);

		return Dot(displacement, displacement) < eps;
	}

	/*-------- vec3 Operations----------*/


	float Dot(const vec3& a, const vec3& b) {
		return a.data[0] * b.data[0] + a.data[1] * b.data[1] + a.data[2] * b.data[2];
	}

	vec3 Cross(const vec3& a, const vec3& b) {
		vec3 result;

		// cross product gives the perpendicular vector to two vectors 

		result.data[0] = a.data[1] * b.data[2] - a.data[2] * b.data[1];
		result.data[1] = a.data[2] * b.data[0] - a.data[0] * b.data[2];
		result.data[2] = a.data[0] * b.data[1] - a.data[1] * b.data[0];

		return result;
	}

	vec3 Normalize(const vec3& a) {

		// get inv magnitude, we will take this scalar and mult to vector, mag will be 1

		float invMagnitude = 1.0f / sqrtf(a.data[0] * a.data[0] + a.data[1] * a.data[1] + a.data[2] * a.data[2]);

		return { a.x / invMagnitude, a.y / invMagnitude, a.z / invMagnitude };
	}

	vec3 Sub(const vec3& a, const vec3& b) {

		return { a.x - b.x, a.y - b.y, a.z - b.z };
	}

	vec3 Add(const vec3& a, const vec3& b) {
		return { a.x + b.x, a.y + b.y, a.z + b.z };
	}

	vec3 Mul(const vec3& a, float scalar) {
		return { a.x * scalar, a.y * scalar, a.z * scalar };
	}


	vec3 Mul(const mat4& m, const vec3& v) {


		bvec3 optV;

		/*
		v.data[0] is multiplied by each element in m.column[0]
		v.data[1] is multiplied by each element in m.column[1]
		v.data[2] is multiplied by each element in m.column[2]
		v.data[3] is multiplied by each element in m.column[3]

		add all vectors resulting vectors from simd ...
		*/

		optV.vector = _mm_fmadd_ps(_mm_set1_ps(v.data[0]), m.column[0],
			_mm_fmadd_ps(_mm_set1_ps(v.data[1]), m.column[1],
				_mm_fmadd_ps(_mm_set1_ps(v.data[2]), m.column[2],
					_mm_mul_ps(_mm_set1_ps(v.data[3]), m.column[3])
				)
			)
		);

		return { optV.x, optV.y, optV.z };
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

		// expanding vec3 for simd operations
		
		bvec3 bNorm(normal);
		bvec3 bInci(incident);
		bvec3 result;
		
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
			bNorm.vector,
			_mm_set1_ps(-2.0f * Dot(incident, bNorm)),
			bInci.vector
		);

		return { result.x, result.y, result.z };
	}

	vec3 Lerp(const vec3& a, const vec3& b, float t) {

		// expanding vec3 for simd operations

		bvec3 optA(a);
		bvec3 optB(b);
		bvec3 result;


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
			_mm_sub_ps(optB.vector, optA.vector),
			_mm_set1_ps(t),
			optA.vector
		);

		return { result.x, result.y, result.z };
	}

	vec3 Slerp(const vec3& a, const vec3& b, float t) {

		// spherical linear interp
		// special case when t<.1, just use linear interp
		// SLERP(a, b, t) = (sin((1-t) * theta) / sin(theta)) * a + (sin(t * theta) / sin(theta)) * b

		if (t < 0.1f) {
			return Lerp(a, b, t);
		}

		// expanding vec3 for simd operations

		bvec3 optA(a);
		bvec3 optB(b);
		bvec3 result;


		float angle = AngleBetweenVectors3(a, b);

		// undefined when angle between two vecs is 0
		if (angle == 0.0f) {
			return { 0.0f,0.0f,0.0f };
		}

		float denominator = sinf(radians(angle));


		result.vector = _mm_fmadd_ps(
			optA.vector,
			_mm_set1_ps(sinf(1 - t) * angle / denominator),
			_mm_mul_ps(optB.vector, _mm_set1_ps(sinf(t * angle) / denominator))
		);

		return { result.x, result.y, result.z };
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


	float AngleBetweenVectors3(const vec3& a, const vec3& b) {

		/*
		consider two vectors a and b

		the |projection| of a onto b is, cos(theta)|a|
		scaling the projection by |b| gives us the dot product being
		dot(a,b) = |a||b|cos(theta)
		solving for theta, we get dot(a,b)/(|a||b|)


		*/

		float denominator = sqrtf(Dot(a, a) * Dot(b, b));

		// case when the magnitude of vector a or vector b is 0 is undefined
		if (denominator == 0.0f) {
			return 0.0f;
		}

		float dot = Dot(a, b);

		return degrees(acosf(dot / denominator));
	}


	float Dot(const bvec3& a, const bvec3& b) {
		return a.data[0] * b.data[0] + a.data[1] * b.data[1] + a.data[2] * b.data[2];
	}

	bvec3 Cross(const bvec3& a, const bvec3& b) {
		bvec3 result;

		// cross product gives the perpendicular vector to two vectors 

		result.data[0] = a.data[1] * b.data[2] - a.data[2] * b.data[1];
		result.data[1] = a.data[2] * b.data[0] - a.data[0] * b.data[2];
		result.data[2] = a.data[0] * b.data[1] - a.data[1] * b.data[0];
		result.data[3] = 0;

		return result;
	}

	bvec3 Normalize(const bvec3& a) {

		// get inv magnitude, we will take this scalar and mult to vector, mag will be 1

		float invMagnitude = 1.0f / sqrtf(a.data[0] * a.data[0] + a.data[1] * a.data[1] + a.data[2] * a.data[2]);

		bvec3 result;

		// _mm_mul_ps used to mult two 128 bit simd registers
		// set1_ps sets a the 4 floats in 128 bit register to said input float
		result.vector = _mm_mul_ps(a.vector, _mm_set1_ps(invMagnitude));

		return result;
	}

	bvec3 Sub(const bvec3& a, const bvec3& b) {

		bvec3 result;

		result.vector = _mm_sub_ps(a.vector, b.vector);

		return result;
	}

	bvec3 Add(const bvec3& a, const bvec3& b) {

		bvec3 result;

		result.vector = _mm_add_ps(a.vector, b.vector);

		return result;
	}

	bvec3 Mul(const bvec3& a, float scalar) {

		bvec3 result;

		result.vector = _mm_mul_ps(a.vector, _mm_set1_ps(scalar));

		return result;

	}


	bvec3 Project(const bvec3& incoming, const bvec3& basis) {
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

	bvec3 Reject(const bvec3& incoming, const bvec3& basis) {
		return Sub(incoming, Project(incoming, basis));
	}

	bvec3 Reflect(const bvec3& incident, const bvec3& normal) {

		bvec3 result;
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

	bvec3 Lerp(const bvec3& a, const bvec3& b, float t) {

		bvec3 result;

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

	bvec3 Slerp(const bvec3& a, const bvec3& b, float t) {

		// spherical linear interp
		// special case when t<.1, just use linear interp
		// SLERP(a, b, t) = (sin((1-t) * theta) / sin(theta)) * a + (sin(t * theta) / sin(theta)) * b

		if (t < 0.1f) {
			return Lerp(a, b, t);
		}

		float angle = AngleBetweenVectors3(a, b);

		// undefined when angle between two vecs is 0
		if (angle == 0.0f) {
			return { 0.0f,0.0f,0.0f };
		}

		float denominator = sinf(radians(angle));


		bvec3 result;

		result.vector = _mm_fmadd_ps(
			a.vector,
			_mm_set1_ps(sinf(1 - t) * angle / denominator),
			_mm_mul_ps(b.vector, _mm_set1_ps(sinf(t * angle) / denominator))
		);

		return result;
	}

	// normalizing lerp vec
	bvec3 Nlerp(const bvec3& a, const bvec3& b, float t) {
		return Normalize(Lerp(a, b, t));
	}


	bool Close(const bvec3& a, const bvec3& b) {

		// get the displacement vec to see how far the terminal
		// points vary from the other, dot product with itself to get the mag
		bvec3 displacement = Sub(a, b);

		return Dot(displacement, displacement) < eps;
	}


	float AngleBetweenVectors3(const bvec3& a, const bvec3& b)
	{
		/*
		consider two vectors a and b

		the |projection| of a onto b is, cos(theta)|a|
		scaling the projection by |b| gives us the dot product being
		dot(a,b) = |a||b|cos(theta)
		solving for theta, we get dot(a,b)/(|a||b|)

		*/
		float denominator = sqrtf(Dot(a, a) * Dot(b, b));

		// case when the magnitude of vector a or vector b is 0 is undefined
		if (denominator == 0.0f) {
			return 0.0f;
		}

		float dot = Dot(a, b);

		return degrees(acosf(dot / denominator));
	}

	/*-------- Vector4 Operations ----------*/

	float Dot(const vec4& a, const vec4& b) {

		vec4 result;
		result.vector = _mm_mul_ps(a.vector, b.vector);

		// _mm_extract_ps maybe ...

		return result.x + result.y + result.z + result.w;
	}



	float AngleBetweenVectors4(const vec4& a, const vec4& b) {

		/*
		consider two vectors a and b

		the |projection| of a onto b is, cos(theta)|a|
		scaling the projection by |b| gives us the dot product being
		dot(a,b) = |a||b|cos(theta)
		solving for theta, we get dot(a,b)/(|a||b|)


		*/
		float denominator = sqrtf(Dot(a, a) * Dot(b, b));

		// case when the magnitude of vector a or vector b is 0 is undefined
		if (denominator == 0.0f) {
			return 0.0f;
		}

		float dot = Dot(a, b);

		return degrees(acosf(dot / denominator));
	}

	vec4 Normalize(const vec4& a) {

		float invMagnitude = 1.0f / sqrtf(
			a.data[0] * a.data[0] + a.data[1] * a.data[1] + a.data[2] * a.data[2] + a.data[3] * a.data[3]
		);

		vec4 result;

		result.vector = _mm_mul_ps(a.vector, _mm_set1_ps(invMagnitude));

		return result;
	}


	vec4 Lerp(const vec4& a, const vec4& b, float t) {

		vec4 result;

		/*

			when t = 0, its a
			t = 0, its b

			lerp(a,b,t) = (1-t)(a) + (t)(b)
			expanding
			a - at + bt

			a + t(b-a)
			t(b-a) + a

		*/

		result.vector = _mm_fmadd_ps(
			_mm_sub_ps(b.vector, a.vector),
			_mm_set1_ps(t),
			a.vector
		);

		return result;
	}

	vec4 Slerp(const vec4& a, const vec4& b, float t) {

		// spherical linear interp
		// special case when t<.1, just use linear interp
		// SLERP(a, b, t) = (sin((1-t) * theta) / sin(theta)) * a + (sin(t * theta) / sin(theta)) * b

		if (t < 0.1f) {
			return Lerp(a, b, t);
		}

		float angle = AngleBetweenVectors4(a, b);
		
		// undefined when angle is 0 between two vectors
		if (angle == 0.0f) {
			return { 0.0f,0.0f,0.0f,0.0f };
		}

		float denominator = sinf(radians(angle));


		vec4 result;

		result.vector = _mm_fmadd_ps(
			a.vector,
			_mm_set1_ps(sinf(1 - t) * angle / denominator),
			_mm_mul_ps(b.vector, _mm_set1_ps(sinf(t * angle) / denominator))
		);

		return result;
	}

	// normalizing lerp vec
	vec4 Nlerp(const vec4& a, const vec4& b, float t) {
		return Normalize(Lerp(a, b, t));
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

	mat4 LookAt(const bvec3& eye, const bvec3& target, const bvec3& up) {

		bvec3 forwardsNorm = Normalize(Sub(target, eye));
		bvec3 rightNorm = Normalize(Cross(forwardsNorm, up));
		bvec3 upNorm = Normalize(Cross(rightNorm, forwardsNorm));
		bvec3 backwardNorm = Mul(forwardsNorm, -1);

		mat4 result;

		result.column[0] = _mm_setr_ps(rightNorm.data[0], upNorm.data[0], backwardNorm.data[0], 0);
		result.column[1] = _mm_setr_ps(rightNorm.data[1], upNorm.data[1], backwardNorm.data[1], 0);
		result.column[2] = _mm_setr_ps(rightNorm.data[2], upNorm.data[2], backwardNorm.data[2], 0);
		result.column[3] = _mm_setr_ps(-Dot(rightNorm, eye), -Dot(upNorm, eye), -Dot(backwardNorm, eye), 1);

		return result;
	}

	mat4 Translation(const vec3& translation) {

		mat4 result;

		result.column_vector[3].x += translation.x;
		result.column_vector[3].y += translation.y;
		result.column_vector[3].z += translation.z;

		result.column[3] = _mm_setr_ps(translation.data[0], translation.data[1], translation.data[2], 1);

		return result;
	}


	mat4 Translation(const bvec3& translation) {

		mat4 result;

		result.column[3] = _mm_setr_ps(translation.data[0], translation.data[1], translation.data[2], 1);

		return result;
	}

	crm::mat4 Translation(const mat4& m, const vec3& translation)
	{
		mat4 result = m;

		result.column_vector[3].x += translation.x;
		result.column_vector[3].y += translation.y;
		result.column_vector[3].z += translation.z;

		return result;
	}

	mat4 Translation(const mat4& m, const bvec3& translation) {

		mat4 result = m;
		
		// just setting the 4th column and adding possible existing translations

		// for bvec3, layout will be <x,y,z,0>, as long as the user doesnt touch the pad
		
		/*
		<x, y, z, 0>    +<tx, ty, tz, w>
			... <tx + x, ty + y, tz + z, 0 + w>

		again, assuming the 0 in the bvec3, if the user doesnt touch the padding
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

		// last two columns are default initalized

		result.column[0] = _mm_setr_ps(cT, -sT, 0, 0);
		result.column[1] = _mm_setr_ps(sT, cT, 0, 0);
		//	result.column[2] = _mm_setr_ps( 0,   0, 1, 0);
		//	result.column[3] = _mm_setr_ps( 0,   0, 0, 1);

		return result;
	}
	

	crm::mat4 Scale(const mat4& m, const vec3& v)
	{
		// upscale to bvec3 to optimize with simd
		bvec3 optV(v);

		mat4 result = m;

		// [m00 * vx, m01 * vx, m02 * vx, m03 * vx]
		// [m10 * vy, m11 * vy, m12 * vy, m13 * vy]
		// [m20 * vz, m21 * vz, m22 * vz, m23 * vz]
		// [m30 * 1 , m31 * 1 , m32 * 1 , m33 * 1 ]

		// setting a scale chunk that multiplies to the two chunks in the input matrix

		__m256 scale_chunk = _mm256_set_ps(1.0f, optV.z, optV.y, optV.x, 1.0f, optV.z, optV.y, optV.x);

		result.chunk[0] = _mm256_mul_ps(m.chunk[0], scale_chunk);
		result.chunk[1] = _mm256_mul_ps(m.chunk[1], scale_chunk);

		return result;
	}

	mat4 Scale(const mat4& m, const bvec3& v) {

		mat4 result = m;

		// [m00 * vx, m01 * vx, m02 * vx, m03 * vx]
		// [m10 * vy, m11 * vy, m12 * vy, m13 * vy]
		// [m20 * vz, m21 * vz, m22 * vz, m23 * vz]
		// [m30 * 1 , m31 * 1 , m32 * 1 , m33 * 1 ]

		// setting a scale chunk that multiplies to the two chunks in the input matrix

		__m256 scale_chunk = _mm256_set_ps(1.0f, v.z, v.y, v.x, 1.0f, v.z, v.y, v.x);

		result.chunk[0] = _mm256_mul_ps(m.chunk[0], scale_chunk);
		result.chunk[1] = _mm256_mul_ps(m.chunk[1], scale_chunk);

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

	mat4 Transpose(const mat4& m) {
		mat4 result;
		__m128 tmp0 = _mm_unpacklo_ps(m.column[0], m.column[1]);
		__m128 tmp1 = _mm_unpackhi_ps(m.column[0], m.column[1]);
		__m128 tmp2 = _mm_unpacklo_ps(m.column[2], m.column[3]);
		__m128 tmp3 = _mm_unpackhi_ps(m.column[2], m.column[3]);

		result.column[0] = _mm_movelh_ps(tmp0, tmp2);
		result.column[1] = _mm_movehl_ps(tmp2, tmp0);
		result.column[2] = _mm_movelh_ps(tmp1, tmp3);
		result.column[3] = _mm_movehl_ps(tmp3, tmp1);

		return result;
	}

	

	// Optimized determinant calculation
	float Determinant(const mat4& m) {
		mat4 A = m;
		float det = 1.0f;

		for (int i = 0; i < 4; ++i) {
			// Get diagonal element as pivot
			float pivot = A.data[i + 4 * i];
			if (fabs(pivot) < eps) return 0.0f;

			det *= pivot;
			// Scale current row by 1/pivot
			__m128 pivotVec = _mm_set1_ps(1.0f / pivot);
			A.column[i] = _mm_mul_ps(A.column[i], pivotVec);

			// Eliminate in all subsequent rows
			for (int j = i + 1; j < 4; ++j) {
				float factor = A.data[i + 4 * j];
				__m128 factorVec = _mm_set1_ps(factor);
				A.column[j] = _mm_sub_ps(A.column[j],
					_mm_mul_ps(factorVec, A.column[i]));
			}
		}

		return det;
	}

	// Combined inverse calculation - eliminates need for separate cofactor/adjugate
	mat4 Inverse(const mat4& m) {
		mat4 result;
		mat4 temp = m;

		// Check if matrix is singular
		if (fabs(Determinant(m)) < eps) {
			return mat4(0.0f,0.0f,0.0f);  // Return zero matrix
		}

		// Gauss-Jordan elimination with SIMD
		for (int i = 0; i < 4; ++i) {
			// Get pivot
			float pivot = temp.data[i + 4 * i];
			__m128 pivotVec = _mm_set1_ps(1.0f / pivot);

			// Scale current row
			temp.column[i] = _mm_mul_ps(temp.column[i], pivotVec);
			result.column[i] = _mm_mul_ps(result.column[i], pivotVec);

			// Eliminate in all other rows
			for (int j = 0; j < 4; ++j) {
				if (j != i) {
					float factor = temp.data[i + 4 * j];
					__m128 factorVec = _mm_set1_ps(factor);
					temp.column[j] = _mm_sub_ps(temp.column[j],
						_mm_mul_ps(factorVec, temp.column[i]));
					result.column[j] = _mm_sub_ps(result.column[j],
						_mm_mul_ps(factorVec, result.column[i]));
				}
			}
		}

		return result;
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

	mat4 Ortho(float l, float r, float t, float b, float f, float n) {


		mat4 result(1.0f);

		result.column_vector[0].x =  2.0f / (r - l); // x scaling factor
		result.column_vector[1].y =  2.0f / (t - b); // y scaling factor 
		result.column_vector[2].z = -2.0f / (f - n); // z scaling factor

		result.column_vector[3].x = -(r + l) / (r - l); // x translation
		result.column_vector[3].y = -(t + b) / (t - b); // y translation
		result.column_vector[3].z = -(f + n) / (f - n); // z translation

		return result;


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