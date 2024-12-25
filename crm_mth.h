#ifndef CHROMA_MATH_H
#define CHROMA_MATH_H

#include <immintrin.h>
#include <intrin.h>

constexpr float pi = 3.14159265359f;
constexpr float eps = 1e-10f;

// used material from getintogamedev on youtube
// https://www.youtube.com/@GetIntoGameDev/videos


/*
 - note, this math library is going to be intended for x64, for 64 bit registers

	when it comes to compilers ... 
	gcc and clang are similar and in their approach to simd ->  both follow same standards an intrinsic sets



	(GCC and Clang)
	Compiler flags:

	For SSE: -msse
	For AVX: -mavx

	MSVC (Microsoft Visual C++)
	Compiler flags:

	For SSE: /arch:SSE
	For AVX: /arch:AVX


 ~ this project is going to be aimed towards said compilers, i will continue development
   and add more accessability for other compilers



*/

namespace crm {

	struct vec2 {
		union {
			float data[2];   

			// anonymous struct to access x and y members
			struct {
				float x, y;
			};
		};
		vec2(float x = 0.0f, float y = 0.0f);
	};

	// batched vec2 ... note -> for simple use cases, it is uncesary to allocate for simd aligment
	// however I included it since you might want to do batched operations with potentially many vec2's
	struct bvec2 {
		union {
			__m128 vector;      // 128-bit SIMD register representing the 3D vector
			float data[4];      // Array of 4 floats, includes 2 for the vector and 2 for padding

			// anonymous struct to access x y and z members
			struct {
				float x, y, _pad0, _pad1;
			};
		};
		bvec2(float x = 0.0f, float y = 0.0f);
	};


	struct vec3 {
		union {
			__m128 vector;      // 128-bit SIMD register representing the 3D vector
			float data[4];      // Array of 4 floats, includes 3 for the vector and 1 for padding 

			struct {
				float x, y, z, _pad0;
			};
		};
		vec3(float x = 0.0f, float y = 0.0f, float z = 0.0f);
	};

	struct vec4 {
		union {
			__m128 vector;      // 128-bit SIMD register representing the 4D vector
			float data[4];      // Array of 4 floats, representing the vector in standard format

			struct {
				float x, y, z, w;
			};
		};
		vec4(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f);
	};

	struct mat4 {
		union {
			__m256 chunk[2];          // Array of two 256-bit SIMD registers (used for AVX operations)
			__m128 column[4];         // Array of four 128-bit SIMD registers (used for SSE operations)
			vec4 column_vector[4];    // Array of 4 vec4, where each element is a column vector of the matrix
			float data[16];           // Array of 16 floats, representing the matrix elements in column-major format
		};
		mat4(float x = 1.0f, float y = 1.0f, float z = 1.0f, float w = 1.0f);
	};

	struct quat {
		union {
			__m128 vector;      // 128-bit SIMD register representing the quaternion
			float data[4];      // Array of 4 floats for x,y,z,w components -> not scalar first

			struct {
				float x, y, z, w;
			};
		};
		quat(float x, float y, float z, float w);
		quat();

		// replaced from MakeQuaternionFromRotation
		quat(float angle, const vec3& axis);

		// replaced from MakeRotationFromVec2Vec
		quat(const vec3& a, const vec3& b);
	};



	// some util functions
	// might add an ifdef for "fast functions" if user doesnt care about slight innacuracies

	/** 
		calculates fast inv sqrt ... refrenced stack over flow, uses a magical number and newtons method, check it out in .cpp file!

		\param x in float
		\returns inverse square of said float
	*/
	float fast_inv_sqrt(float x);

	/** 
		calculates sqrt ...
		based on fast inv square

		\param x in float
		\returns returns 1/fast_inv_sqrt(x)
	*/
	float fast_sqrt(float x);

	/** 
		truncates float to int ...

		\param f in float
		\returns returns int trunc of float
	*/

	int fast_ftoi(float f);


	/**
		Convert from degrees to radians.

		\param angle the angle to convert (in degrees)
		\returns the angle converted to radians
	*/
	float radians(float angle);

	/**
		Convert from radians to degrees.

		\param angle the angle to convert (in radians)
		\returns the angle converted to degrees
	*/
	float degrees(float angle);

	/*-------- Vec3 Operations    ----------*/

	/**
		Construct a vec3 from individual floating point components.
		old, now there is a constructor for vec3
	*/
	//vec3 MakeVec3(float x, float y, float z);

	/**
		Compute the dot product between two vec3s,
		as with any dot product, the order doesn't matter.

		\param a the first vector
		\param b the second vector
		\returns the dot product: a.b
	*/
	float Dot(const vec3& a, const vec3& b);

	/**
		Compute the cross product between two vec3s

		\param a the first vector
		\param b the second vector
		\returns a new vector storing the cross product: c = axb
	*/
	vec3 Cross(const vec3& a, const vec3& b);

	/**
		\param a the vector to normalize, cannot have zero magnitude
		\returns a new vec3, being parallel with the input a, having unit length.
	*/
	vec3 Normalize(const vec3& a);

	/**
		Compute a vector subtraction.

		\param a the first vector
		\param b the second vector
		\returns a new vector storing the difference: c = a - b
	*/
	vec3 Sub(const vec3& a, const vec3& b);

	/**
		Compute a vector addition.

		\param a the first vector
		\param b the second vector
		\returns a new vector storing the sum: c = a + b
	*/
	vec3 Add(const vec3& a, const vec3& b);

	/**
		Compute a vector scalar multiplication.

		\param a the vector
		\param scalar the scalar
		\returns a new vector storing the scaled vector: c = scalar * a
	*/
	vec3 Mul(const vec3& a, float scalar);

	/**
		Get the angle between two vectors.

		\param a the first vector, cannot have zero magnitude
		\param b the second vector, cannot have zero magnitude
		\returns the angle between the vectors a & b, in degrees
	*/
	float AngleBetweenVectors3(const vec3& a, const vec3& b);

	/**
		Get the projection of one vector onto another.
		Any vector v can be decomposed with regard to another vector u:

			v	= v(parallel with u) + v(perpendicular with u)
				= projection(v onto u) + rejection(v onto u)

		\param incoming the vector to be projected
		\param basis the vector onto which to be projected, cannot have zero magnitude
		\returns a new vector, parallel with basis, storing the vector projection of incoming onto basis
	*/
	vec3 Project(const vec3& incoming, const vec3& basis);

	/**
		Get the rejection of one vector onto another.
		Any vector v can be decomposed with regard to another vector u:

			v	= v(parallel with u) + v(perpendicular with u)
				= projection(v onto u) + rejection(v onto u)

		\param incoming the vector to be rejected
		\param basis the vector to do the rejecting, cannot have zero magnitude
		\returns a new vector, orthogonal to basis, storing the vector rejection of incoming from basis
	*/
	vec3 Reject(const vec3& incoming, const vec3& basis);

	/**
		Compute a vector reflection.

		\param incident a direction vector incident to (pointing towards) the point of impact.
		\param normal the normal vector about which to reflect. Must have unit length.
		\returns a new vector representing the direction after reflecting.
	*/
	vec3 Reflect(const vec3& incident, const vec3& normal);

	/**
		Linearly interpolate between two vectors.

		\param a the first vector
		\param b the second vector
		\param t the interpolation parameter. Typically between 0 and 1, though this isn't enforced
		\returns a new vector, being a linear interpolation between a and b.
	*/
	vec3 Lerp(const vec3& a, const vec3& b, float t);

	/**
		Spherical Linear interpolation between two vectors.
		lerp will take a straight line between vectors, on the other hand,
		slerp interpolates angle-wise, in a rotational sense.

		\param a the first vector, should be normalized.
		\param b the second vector, should be normalized.
		\param t the interpolation parameter. Typically between 0 and 1, though this isn't enforced
		\returns a new vector, being a linear interpolation between a and b.
	*/
	vec3 Slerp(const vec3& a, const vec3& b, float t);

	/**
		Normalized Linear interpolation between two vectors.
		Normalizing the result of lerp will approximate slerp.

		\param a the first vector, should be normalized.
		\param b the second vector, should be normalized.
		\param t the interpolation parameter. Typically between 0 and 1, though this isn't enforced
		\returns a new vector, being a linear interpolation between a and b.
	*/
	vec3 Nlerp(const vec3& a, const vec3& b, float t);

	/**
		ease in takes t and makes it increase slowly with then speeds up
		\param t the "time" value to interpolate with
		\returns adjusted "time" value
	*/
	float ease_in(float t);

	/**
		ease out takes t and makes it increased fast then slows down
		\param t the "time" value to interpolate with
		\returns adjusted "time" value
	*/
	float ease_out(float t);

	/**
		Indicates whether two vectors are within epsilon of one another.
	*/
	bool Close(const vec3& a, const vec3& b);

	/*-------- Vector4 Operations ----------*/

	/*
		Under the hood, vec3 is really a vec4 with zeroed out w component,
		however as convention has it right now, it's being treated as a seperate type.
	*/

	/**
		\returns a normalized copy of the given vector.
	*/
	vec4 Normalize(const vec4& a);

	/**
		\returns the dot product result = a.b
	*/
	float Dot(const vec4& a, const vec4& b);

	/*-------- Matrix4 Operations ----------*/

	/**
		\ old, now there is constructor for mat4
	*/
	//mat4 MakeIdentity4();

	/**
		Make a perspective projection matrix.

		\param fovy the field of view angle of the frustrum (in degrees)
		\param aspect the aspect ratio width/height
		\param near the near view distance of the frustrum
		\param far the far distance of the frustrum
		\returns a new mat4 representing the perspective projection transform
	*/
	mat4 MakePerspectiveProjection(float fovy, float aspect, float near, float far);

	/**
		Make a view matrix (translates and rotates the world around the 
		given reference point)

		\param eye the position of the viewer
		\param target the position the viewer is looking at
		\param up the up direction from the viewer's point of reference
		\returns a new mat4 representing the view transform
	*/
	mat4 LookAt(const vec3& eye, const vec3& target, const vec3& up);

	/**
		Make a translation transform matrix.

		\param translation the displacement to apply
		\returns a new mat4 representing the transform
	*/
	mat4 Translation(const vec3& translation);

	/**
	    Translate an input matrix.

	\param translation the displacement to apply
	\returns a new mat4 representing the transformed input mat4
	*/
	mat4 Translation(const mat4& m, const vec3& translation);

	/**
		Make a rotation around the x-axis.

		\param angle the angle to rotate by (in degrees)
		\returns a new mat4 representing the transform
	*/
	mat4 XRotation(float angle);

	/**
		Make a rotation around the y-axis.

		\param angle the angle to rotate by (in degrees)
		\returns a new mat4 representing the transform
	*/
	mat4 YRotation(float angle);

	/**
		Make a rotation around the z-axis.

		\param angle the angle to rotate by (in degrees)
		\returns a new mat4 representing the transform
	*/
	mat4 ZRotation(float angle);

	/**
		Transform a vector by a matrix.

		\param m the matrix to apply
		\param v the vector to transform
		\returns a new vec4 representing the matrix multiplication: result = m*v
	*/
	vec4 Mul(const mat4& m, const vec4& v);

	/**
		Multiply two matrices

		\param m1 the original matrix
		\param m2 the new matrix to multiply onto m1
		\returns a new mat4 representing the matrix multiplication: m3 = m2*m1
	*/
	mat4 Mul(const mat4& m2, const mat4& m1);

	/**
		\returns the matrix sum m3 = m1 + m2
	*/
	mat4 Add(const mat4& m1, const mat4& m2);

	/**
		\returns the scalar multiplication result = scalar * matrix
	*/
	mat4 Mul(const mat4& matrix, float scalar);

	/**
		Blend (linearly interpolate) two matrices.

		\param m1 the start matrix (t = 0)
		\param m2 the end matrix (t = 1)
		\param t the interpolation parameter
		\returns the result m3 = m1 + t * (m2 - m1)
	*/
	mat4 Lerp(const mat4& m1, const mat4& m2, float t);

	/**
		\returns a transposed copy of the given matrix
	*/
	mat4 Transpose(const mat4& matrix);

	/**
		Compute a transform matrix inverse.

		General matrix inverses are computationally intense, however
		most transform matrices can be expressed in the form:

		M = (aX | bY | cZ | I)

		where:

			a: x axis scaling factor
			X: forwards rotation vector
			b: y axis scaling factor
			Y: right rotation vector
			c: z axis scale factor
			Z: up rotation vector
			I: [0 0 0 1]

		Matrices in this form have a closed form inverse.

		Source: https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html

		\param matrix the matrix to invert
		\returns the inverse
	*/
	mat4 TransformInverse(const mat4& matrix);

	/*-------- Quaternion Operations ----------*/

	/**
		\returns a quaternion made from individual components.
		// old, constructor for quaternion exists 
	*/
	//quat MakeQuaternionFromComponents(float x, float y, float z, float w);

	/**
		Make a quaternion from a rotation operation.

		\param angle the rotation angle (in degrees)
		\param axis the axis of rotation
		\returns the corresponding quaternion

		old : constructor for this
	*/
	// quat MakeQuaternionFromRotation(float angle, vec3 axis);


	 /**
 		Make a quaternion tracking a rotation from vector a to vector b.

		old : constructor for this
	 */
	// quat MakeRotationFromVec2Vec(vec3 a, vec3 b);

	/**
		\returns the quaternion's axis of rotation
	*/
	vec3 GetAxisFromQuaternion(const quat& q);

	/**
		\returns the quaternion's angle, in degrees
	*/
	float GetAngleFromQuaternion(const quat& q);

	/**
		\returns the sum of two quaternions
	*/
	quat Add(const quat& q1, const quat& q2);

	/**
		\returns the difference of two quaternions
	*/
	quat Sub(const quat& q1, const quat& q2);

	/**
		\returns a scaled copy of the quaternion
	*/
	quat Mul(const quat& q, float scalar);

	/**
		\returns the dot product of two quaternions
	*/
	float Dot(const quat& q1, const quat& q2);

	/**
		\returns whether two quaternions are sufficiently close.
	*/
	bool Close(const quat& q1, const quat& q2);

	/**
		It's possible for two quaternions to be the same, but mathematically different.
		Ie. a rotation in the opposite angle through the opposite axis is actually still the same.

		\returns whether two quaternions have the same orientation.
	*/
	bool QuatSameOrientation(const quat& q1, const quat& q2);

	/**
		\returns a normalized quaternion
	*/
	quat Normalize(const quat& q);

	/**
		\returns the conjugate quaternion, a rotation of the same angle, around the opposite axis.
	*/
	quat GetConjQuat(const quat& q);

	/**
		\returns the inverse of the given quaternion.
	*/
	quat InvQuat(const quat& q);


}


#endif