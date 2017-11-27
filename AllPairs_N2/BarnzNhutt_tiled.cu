#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include "Constants.h"
#include <cuda.h>
#define TILE 1024

void initializeBodies(float4* pos, float4* vel);
void runSimulation(float4* pos, float4* vel, char* image, float* hdImage);
__global__ void interactBodies(float4* pos, float4* vel);
__host__ __device__ float magnitude(vec3 v);
void createFrame(char* image, float* hdImage, float4* pos, float4* vel, int step);
float toPixelSpace(float p, int size);
void renderClear(char* image, float* hdImage);
void renderBodies(float4* pos, float4* vel, float* hdImage);
void colorDot(float x, float y, float vMag, float* hdImage);
void colorAt(int x, int y, struct color c, float f, float* hdImage);
unsigned char colorDepth(unsigned char x, unsigned char p, float f);
float clamp(float x);
void writeRender(char* data, float* hdImage, int step);
__device__ float4 bodyBodyInteraction(float4 posi, float4 posj, float4 Fi);
__device__ float4 tile_calculation(float4 pos, float4 Fi);
__global__ void calculate_forces(float4 *pos, float4* acc);
__global__ void update(float4* pos, float4* vel, float4* Fi);

int main()
{
	std::cout << SYSTEM_THICKNESS << "AU thick disk\n";;
	char *image = new char[WIDTH*HEIGHT*3];
	float *hdImage = new float[WIDTH*HEIGHT*3];
	//struct body *bodies = new struct body[NUM_BODIES];
	float4* pos = new float4[NUM_BODIES];
	float4* vel = new float4[NUM_BODIES];
	initializeBodies(pos,vel);
	runSimulation(pos,vel, image, hdImage);
	std::cout << "\nwe made it\n";
	delete[] image;
	return 0;
}

void initializeBodies(float4* pos, float4* vel)
{
	using std::uniform_real_distribution;
	uniform_real_distribution<float> randAngle (0.0, 200.0*PI);
	uniform_real_distribution<float> randRadius (INNER_BOUND, SYSTEM_SIZE);
	uniform_real_distribution<float> randHeight (0.0, SYSTEM_THICKNESS);
	std::default_random_engine gen (0);
	float angle;
	float radius;
	float velocity;

	//STARS
	velocity = 0.67*sqrt((G*SOLAR_MASS)/(4*BINARY_SEPARATION*TO_METERS));
	//STAR 1
	pos[0].x = 0.0;///-BINARY_SEPARATION;
	pos[0].y = 0.0;
	pos[0].z = 0.0;
	vel[0].x = 0.0;
	vel[0].y = 0.0;//velocity;
	vel[0].z = 0.0;
	vel[0].w = SOLAR_MASS;

	    ///STARTS AT NUMBER OF STARS///
	float totalExtraMass = 0.0;
	for (int index=1; index<NUM_BODIES; index++)
	{
		angle = randAngle(gen);
		radius = sqrt(SYSTEM_SIZE)*sqrt(randRadius(gen));
		velocity = pow(((G*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
					  	  	  	  	  / (radius*TO_METERS)), 0.5);
		pos[index].x =  radius*cos(angle);
		pos[index].y =  radius*sin(angle);
		pos[index].z =  randHeight(gen)-SYSTEM_THICKNESS/2;
		vel[index].x =  velocity*sin(angle);
		vel[index].y = -velocity*cos(angle);
		vel[index].z =  0.0;
		vel[index].w = (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
		pos[index].w = vel[index].w;
		totalExtraMass += (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
	}
	std::cout << "\nTotal Disk Mass: " << totalExtraMass;
	std::cout << "\nEach Particle weight: " << (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES
			  << "\n______________________________\n";
}

void runSimulation(float4* pos, float4* vel, char* image, float* hdImage)
{
	createFrame(image, hdImage, pos,vel, 1);
	float4 *d_pos;
	float4 *d_vel;
	float4 *d_acc;
	cudaMalloc(&d_pos,NUM_BODIES*sizeof(float4));
	cudaMalloc(&d_vel,NUM_BODIES*sizeof(float4));
	cudaMalloc(&d_acc,NUM_BODIES*sizeof(float4));
	
	int nBlocks=(NUM_BODIES+TILE-1)/TILE;
	for (int step=1; step<STEP_COUNT; step++)
	{
		std::cout << "\nBeginning timestep: " << step;
		printf("\nStartK\n");
		cudaMemcpy(d_pos, pos, NUM_BODIES*sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vel, vel, NUM_BODIES*sizeof(float4), cudaMemcpyHostToDevice);
		printf("StartK1\n");	
		//interactBodies<<<nBlocks,1024>>>(d_pos,d_vel);
		calculate_forces<<<nBlocks,TILE,TILE*sizeof(float4)>>>(d_pos,d_acc);
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if(error!=cudaSuccess)
		{
			printf("CUDA error:%s\n",cudaGetErrorString(error));
		}
		update<<<nBlocks,TILE>>>(d_pos,d_vel,d_acc);
		cudaDeviceSynchronize();
		error=cudaGetLastError();
		if(error!=cudaSuccess)
			printf("CUDA error:%s\n",cudaGetErrorString(error));
		printf("EndK\n");
		cudaMemcpy( pos,d_pos, NUM_BODIES*sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemcpy( vel, d_vel,NUM_BODIES*sizeof(float4), cudaMemcpyDeviceToHost);
		printf("EndK2\n");

		if (step%RENDER_INTERVAL==0)
		{
			createFrame(image, hdImage, pos, vel, step + 1);
		}
		if (DEBUG_INFO) {std::cout << "\n-------Done------- timestep: "
			       << step << "\n" << std::flush;}
	}
}
__global__ void interactBodies(float4* pos, float4* vel)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < NUM_BODIES)
	{	
		float Fx=0.0f; float Fy=0.0f; float Fz=0.0f;
		for(int tile=0; tile<gridDim.x; tile++)
		{
			__shared__ float4 spos[1024];
			spos[threadIdx.x]=pos[tile*blockDim.x+threadIdx.x];
			__syncthreads();
			
			for(int j=0; j < 1024; j++)
			{
				if(i!=tile*1024+j){
					vec3 posDiff;
					posDiff.x = (pos[i].x-spos[j].x)*TO_METERS;
					posDiff.y = (pos[i].y-spos[j].y)*TO_METERS;
					posDiff.z = (pos[i].z-spos[j].z)*TO_METERS;
					float dist = sqrt(posDiff.x*posDiff.x+posDiff.y*posDiff.y+posDiff.z*posDiff.z);
					float F = TIME_STEP*(G*vel[i].w*vel[j].w) / ((dist*dist + SOFTENING*SOFTENING) * dist);
					float Fa = F/vel[i].w;
					Fx-=Fa*posDiff.x;
					Fy-=Fa*posDiff.y;
					Fz-=Fa*posDiff.z;
				}
			}
			__syncthreads();
		}	
		vel[i].x += Fx;
		vel[i].y += Fy;
		vel[i].z += Fz;
		pos[i].x += TIME_STEP*vel[i].x/TO_METERS;
		pos[i].y += TIME_STEP*vel[i].y/TO_METERS;
		pos[i].z += TIME_STEP*vel[i].z/TO_METERS;
	}
}

__device__ float4 bodyBodyInteraction(float4 posi, float4 posj, float4 Fi)
{
	float4 r;
	r.x = (posj.x - posi.x)*TO_METERS;
	r.y = (posj.y - posi.y)*TO_METERS;
	r.z = (posj.z - posi.z)*TO_METERS;
	float dist = sqrt(r.x*r.x+r.y*r.y+r.z*r.z);
	float F = TIME_STEP*(G*posi.w) / ((dist*dist + SOFTENING*SOFTENING) * dist);
	Fi.x -= F*r.x;
	Fi.y -= F*r.y;
	Fi.z -= F*r.z;
	return Fi;	
}

__device__ float4 tile_calculation(float4 pos, float4 Fi)
{
	int i;
	extern __shared__ float4 shPosition[];
	for(i=0; i<blockDim.x; i++)
	{
		Fi=bodyBodyInteraction(pos,shPosition[i],Fi);
	}
	return Fi;
}

__global__ void calculate_forces(float4 *pos, float4 *acc)
{
	extern __shared__ float4 shPosition[];
	float4 mypos;
	int i, tile;
	float4 Fi = {0.0f, 0.0f, 0.0f,0.0f};
	int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	if(gtid<NUM_BODIES)
	{
		mypos=pos[gtid];
		for(i=0, tile=0; i<NUM_BODIES; i+=TILE, tile++)
		{
			int idx=tile*blockDim.x + threadIdx.x;
			shPosition[threadIdx.x]=pos[idx];
			__syncthreads();
			if(gtid!=idx)
				Fi=tile_calculation(mypos,Fi);
			__syncthreads();
		}
		//save to global mem for integration step
		//printf("%f %f %f\n",Fi.x, Fi.y, Fi.z);
		acc[gtid]=Fi;
	}
}

__global__ void update(float4* pos, float4* vel, float4* Fi)
{
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if(i<NUM_BODIES)
	{
		vel[i].x += Fi[i].x;
		vel[i].y += Fi[i].y;
		vel[i].z += Fi[i].z;
		pos[i].x += TIME_STEP*vel[i].x/TO_METERS;
		pos[i].y += TIME_STEP*vel[i].y/TO_METERS;
		pos[i].z += TIME_STEP*vel[i].z/TO_METERS;
	}
}

__host__ __device__ float magnitude(vec3 v)
{
	return sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
}

void createFrame(char* image, float* hdImage, float4* pos, float4* vel, int step)
{
	std::cout << "\nWriting frame " << step;
	if (DEBUG_INFO)	{std::cout << "\nClearing Pixels..." << std::flush;}
	renderClear(image, hdImage);
	if (DEBUG_INFO) {std::cout << "\nRendering Particles..." << std::flush;}
	renderBodies(pos, vel, hdImage);
	if (DEBUG_INFO) {std::cout << "\nWriting frame to file..." << std::flush;}
	writeRender(image, hdImage, step);
}

void renderClear(char* image, float* hdImage)
{
	for (int i=0; i<WIDTH*HEIGHT*3; i++)
	{
	//	char* current = image + i;
		image[i] = 0; //char(image[i]/1.2);
		hdImage[i] = 0.0;
	}
}

void renderBodies(float4* pos, float4* vel, float* hdImage)
{
	/// ORTHOGONAL PROJECTION
	for(int i=0; i<NUM_BODIES; i++)
	{

		int x = toPixelSpace(pos[i].x, WIDTH);
		int y = toPixelSpace(pos[i].y, HEIGHT);

		if (x>DOT_SIZE && x<WIDTH-DOT_SIZE &&
			y>DOT_SIZE && y<HEIGHT-DOT_SIZE)
		{
			float vxsqr=vel[i].x*vel[i].x;
			float vysqr=vel[i].y*vel[i].y;
			float vzsqr=vel[i].z*vel[i].z;
			float vMag = sqrt(vxsqr+vysqr+vzsqr);
			colorDot(pos[i].x, pos[i].y, vMag, hdImage);
		}
	}
}

float toPixelSpace(float p, int size)
{
	return (size/2.0)*(1.0+p/(SYSTEM_SIZE*RENDER_SCALE));
}

void colorDot(float x, float y, float vMag, float* hdImage)
{
	const float velocityMax = MAX_VEL_COLOR; //35000
	const float velocityMin = sqrt(0.8*(G*(SOLAR_MASS+EXTRA_MASS*SOLAR_MASS))/
			(SYSTEM_SIZE*TO_METERS)); //MIN_VEL_COLOR;
	const float vPortion = sqrt((vMag-velocityMin) / velocityMax);
	color c;
	c.r = clamp(4*(vPortion-0.333));
	c.g = clamp(fmin(4*vPortion,4.0*(1.0-vPortion)));
	c.b = clamp(4*(0.5-vPortion));
	for (int i=-DOT_SIZE/2; i<DOT_SIZE/2; i++)
	{
		for (int j=-DOT_SIZE/2; j<DOT_SIZE/2; j++)
		{
			float xP = floor(toPixelSpace(x, WIDTH));
			float yP = floor(toPixelSpace(y, HEIGHT));
			float cFactor = PARTICLE_BRIGHTNESS /
					(pow(exp(pow(PARTICLE_SHARPNESS*
					(xP+i-toPixelSpace(x, WIDTH)),2.0))
				       + exp(pow(PARTICLE_SHARPNESS*
					(yP+j-toPixelSpace(y, HEIGHT)),2.0)),/*1.25*/0.75)+1.0);
			colorAt(int(xP+i),int(yP+j),c, cFactor, hdImage);
		}
	}

}

void colorAt(int x, int y, struct color c, float f, float* hdImage)
{
	int pix = 3*(x+WIDTH*y);
	hdImage[pix+0] += c.r*f;//colorDepth(c.r, image[pix+0], f);
	hdImage[pix+1] += c.g*f;//colorDepth(c.g, image[pix+1], f);
	hdImage[pix+2] += c.b*f;//colorDepth(c.b, image[pix+2], f);
}

unsigned char colorDepth(unsigned char x, unsigned char p, float f)
{
	return fmax(fmin((x*f+p),255),0);
//	unsigned char t = fmax(fmin((x*f+p),255),0);
//	return 2*t-(t*t)/255;
}

float clamp(float x)
{
	return fmax(fmin(x,1.0),0.0);
}

void writeRender(char* data, float* hdImage, int step)
{
	
	for (int i=0; i<WIDTH*HEIGHT*3; i++)
	{
		data[i] = int(255.0*clamp(hdImage[i]));
	}

	int frame = step/RENDER_INTERVAL + 1;//RENDER_INTERVAL;
	std::string name = "images/Step"; 
	int i = 0;
	if (frame == 1000) i++; // Evil hack to avoid extra 0 at 1000
	for (i; i<4-floor(log(frame)/log(10)); i++)
	{
		name.append("0");
	}
	name.append(std::to_string(frame));
	name.append(".ppm");

	std::ofstream file (name, std::ofstream::binary);

	if (file.is_open())
	{
//		size = file.tellg();
		file << "P6\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";
		file.write(data, WIDTH*HEIGHT*3);
		file.close();
	}

}

