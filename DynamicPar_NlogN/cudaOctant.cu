#include "Constants.h"

class Octant
{
public:
	vec3 *mid;
	double length;
	__host__ __device__ Octant(vec3* m, double l)
	{
		mid = m;
		length = l;
	}
	__host__ __device__ Octant()
	{
		length=0.0;
	}

	__host__ __device__ ~Octant()
	{
		delete mid;
	}

	__host__ __device__ double getLength()
	{
		return length;
	}
	
	__host__ __device__ vec3* getMid()
	{
		return mid;
	}

	__host__ __device__ inline bool contains(vec3 p)
	{
		return p.x<=mid->x+length/2.0 && p.x>=mid->x-length/2.0 &&
			   p.y<=mid->y+length/2.0 && p.y>=mid->y-length/2.0 &&
			   p.z<=mid->z+length/2.0 && p.z>=mid->z-length/2.0;
	}

	__host__ __device__ Octant* mUNW()
	{
		vec3 *newMid = new struct vec3;
		double newL = length/4.0;
		newMid->x = mid->x-newL;
		newMid->y = mid->y+newL;
		newMid->z = mid->z+newL;
		return new Octant(newMid, length/2.0);
	}

	__host__ __device__ Octant* mUNE()
	{
		vec3 *newMid = new struct vec3;
		double newL = length/4.0;
		newMid->x = mid->x+newL;
		newMid->y = mid->y+newL;
		newMid->z = mid->z+newL;
		return new Octant(newMid, length/2.0);
	}

	__host__ __device__ Octant* mUSW()
	{
		vec3 *newMid = new struct vec3;
		double newL = length/4.0;
		newMid->x = mid->x-newL;
		newMid->y = mid->y-newL;
		newMid->z = mid->z+newL;
		return new Octant(newMid, length/2.0);
	}

	__host__ __device__ Octant* mUSE()
	{
		vec3 *newMid = new struct vec3;
		double newL = length/4.0;
		newMid->x = mid->x+newL;
		newMid->y = mid->y-newL;
		newMid->z = mid->z+newL;
		return new Octant(newMid, length/2.0);
	}

	__host__ __device__ Octant* mDNW()
	{
		vec3 *newMid = new struct vec3;
		double newL = length/4.0;
		newMid->x = mid->x-newL;
		newMid->y = mid->y+newL;
		newMid->z = mid->z-newL;
		return new Octant(newMid, length/2.0);
	}

	__host__ __device__ Octant* mDNE()
	{
		vec3 *newMid = new struct vec3;
		double newL = length/4.0;
		newMid->x = mid->x+newL;
		newMid->y = mid->y+newL;
		newMid->z = mid->z-newL;
		return new Octant(newMid, length/2.0);
	}

	__host__ __device__ Octant* mDSW()
	{
		vec3 *newMid = new struct vec3;
		double newL = length/4.0;
		newMid->x = mid->x-newL;
		newMid->y = mid->y-newL;
		newMid->z = mid->z-newL;
		return new Octant(newMid, length/2.0);
	}

	__host__ __device__ Octant* mDSE()
	{
		vec3 *newMid = new struct vec3;
		double newL = length/4.0;
		newMid->x = mid->x+newL;
		newMid->y = mid->y-newL;
		newMid->z = mid->z-newL;
		return new Octant(newMid, length/2.0);
	}

};

