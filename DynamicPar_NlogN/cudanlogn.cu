#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include <omp.h>
#include "cudaBhtree.cu"
#include "Constants.h"

void initializeBodies(struct body* bods);
void runSimulation(struct body* b, struct body* db, char* image, double* hdImage);
__global__ void interactBodies(struct body* b);
__device__ void singleInteraction(struct body* a, struct body* b);
__host__ __device__ double magnitude(vec3 v);
__device__ void updateBodies(struct body* b);
void createFrame(char* image, double* hdImage, struct body* b, int step);
double toPixelSpace(double p, int size);
void renderClear(char* image, double* hdImage);
void renderBodies(struct body* b, double* hdImage);
void colorDot(double x, double y, double vMag, double* hdImage);
void colorAt(int x, int y, struct color c, double f, double* hdImage);
unsigned char colorDepth(unsigned char x, unsigned char p, double f);
double clamp(double x);
__global__ void recinsert(Bhtree *tree, body* insertBod);
void writeRender(char* data, double* hdImage, int step);
__device__ inline bool contains(Octant *root, vec3 p);
__global__ void interactInTree(Bhtree *tree, struct body* b);
__host__ __device__ double magnitude(float x, float y, float z);

int main()
{
	std::cout << SYSTEM_THICKNESS << "AU thick disk\n";;
	char *image = new char[WIDTH*HEIGHT*3];
	double *hdImage = new double[WIDTH*HEIGHT*3];
	struct body *bodies = new struct body[NUM_BODIES];
	struct body *d_bodies;
	cudaMalloc(&d_bodies,NUM_BODIES*sizeof(struct body));
	
	initializeBodies(bodies);
	cudaMemcpy(d_bodies,bodies,NUM_BODIES*sizeof(struct body),cudaMemcpyHostToDevice);
	runSimulation(bodies, d_bodies, image, hdImage);
	std::cout << "\nwe made it\n";
	delete[] bodies;
	delete[] image;
	return 0;
}

void initializeBodies(struct body* bods)
{
	using std::uniform_real_distribution;
	uniform_real_distribution<double> randAngle (0.0, 200.0*PI);
	uniform_real_distribution<double> randRadius (INNER_BOUND, SYSTEM_SIZE);
	uniform_real_distribution<double> randHeight (0.0, SYSTEM_THICKNESS);
	std::default_random_engine gen (0);
	double angle;
	double radius;
	double velocity;
	struct body *current;

	//STARS
	velocity = 0.67*sqrt((G*SOLAR_MASS)/(4*BINARY_SEPARATION*TO_METERS));
	//STAR 1
	current = &bods[0];
	current->position.x = 0.0;///-BINARY_SEPARATION;
	current->position.y = 0.0;
	current->position.z = 0.0;
	current->velocity.x = 0.0;
	current->velocity.y = 0.0;//velocity;
	current->velocity.z = 0.0;
	current->mass = SOLAR_MASS;

	    ///STARTS AT NUMBER OF STARS///
	double totalExtraMass = 0.0;
	for (int index=1; index<NUM_BODIES; index++)
	{
		angle = randAngle(gen);
		radius = sqrt(SYSTEM_SIZE)*sqrt(randRadius(gen));
		velocity = pow(((G*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
					  	  	  	  	  / (radius*TO_METERS)), 0.5);
		current = &bods[index];
		current->position.x =  radius*cos(angle);
		current->position.y =  radius*sin(angle);
		current->position.z =  randHeight(gen)-SYSTEM_THICKNESS/2;
		current->velocity.x =  velocity*sin(angle);
		current->velocity.y = -velocity*cos(angle);
		current->velocity.z =  0.0;
		current->mass = (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
		totalExtraMass += (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
	}
	std::cout << "\nTotal Disk Mass: " << totalExtraMass;
	std::cout << "\nEach Particle weight: " << (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES
			  << "\n______________________________\n";
}

void runSimulation(struct body* b, struct body* db, char* image, double* hdImage)
{
	createFrame(image, hdImage, b, 1);
	printf("here (done with createFrame1)");
	for (int step=1; step<STEP_COUNT; step++)
	{
		std::cout << "\nBeginning timestep: " << step;
		interactBodies<<<1,1>>>(db);
		cudaError_t error = cudaGetLastError();
                        if(error!=cudaSuccess)
                                printf("\nCUDA error:%s",cudaGetErrorString(error));
		cudaMemcpy(b,db,NUM_BODIES*sizeof(struct body),cudaMemcpyDeviceToHost);
		if (step%RENDER_INTERVAL==0)
		{
			createFrame(image, hdImage, b, step + 1);
		}
		if (DEBUG_INFO) {std::cout << "\n-------Done------- timestep: "
			       << step << "\n" << std::flush;}
	}
}

__device__ Bhtree *Gtree;

__global__ void recinsert(Bhtree *tree, body* insertBod)
{
	const int num_streams=8;
	cudaStream_t streams[num_streams];

	for(int i=0; i<num_streams; i++)
		cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
		
	if (tree->myBod==NULL)
	{
		tree->myBod = insertBod;
	} else //if (!isExternal())
	{
		bool isExtern = tree->UNW==NULL && tree->UNE==NULL && tree->USW==NULL && tree->USE==NULL;
		isExtern = isExtern && tree->DNW==NULL && tree->DNE==NULL && tree->DSW==NULL && tree->DSE==NULL;
		body *updatedBod;
		if (!isExtern)
		{
			updatedBod = new struct body;
			updatedBod->position.x = (insertBod->position.x*insertBod->mass +
							       tree->myBod->position.x*tree->myBod->mass) /
							  (insertBod->mass+tree->myBod->mass);
			updatedBod->position.y = (insertBod->position.y*insertBod->mass +
								   tree->myBod->position.y*tree->myBod->mass) /
							  (insertBod->mass+tree->myBod->mass);
			updatedBod->position.z = (insertBod->position.z*insertBod->mass +
								   tree->myBod->position.z*tree->myBod->mass) /
							  (insertBod->mass+tree->myBod->mass);
			updatedBod->mass = insertBod->mass+tree->myBod->mass;
		//	delete myBod;
			if (tree->toDelete!=NULL) delete tree->toDelete;
			tree->toDelete = updatedBod;
			tree->myBod = updatedBod;
			updatedBod = insertBod;
		} else {
			updatedBod = tree->myBod;
		}
		Octant *unw = tree->octy->mUNW();
		if (contains(unw,updatedBod->position))
		{
			if (tree->UNW==NULL) { tree->UNW = new Bhtree(unw); }
			else { delete unw; }
			recinsert<<<1,1,0,streams[0]>>>(tree->UNW,updatedBod);
		} else {
			delete unw;
			Octant *une = tree->octy->mUNE();
			if (contains(une,updatedBod->position))
			{
				if (tree->UNE==NULL) { tree->UNE = new Bhtree(une); }
				else { delete une; }
				recinsert<<<1,1,0,streams[1]>>>(tree->UNE,updatedBod);
			} else {
				delete une;
				Octant *usw = tree->octy->mUSW();
				if (contains(usw,updatedBod->position))
				{
					if (tree->USW==NULL) { tree->USW = new Bhtree(usw); }
					else { delete usw; }
					recinsert<<<1,1,0,streams[2]>>>(tree->USW,updatedBod);
				} else {
					delete usw;
					Octant *use = tree->octy->mUSE();
					if (contains(use,updatedBod->position))
					{
						if (tree->USE==NULL) { tree->USE = new Bhtree(use); }
						else { delete use; }
						recinsert<<<1,1,0,streams[3]>>>(tree->USE,updatedBod);
					} else {
						delete use;
						Octant *dnw = tree->octy->mDNW();
						if (contains(dnw,updatedBod->position))
						{
							if (tree->DNW==NULL) { tree->DNW = new Bhtree(dnw); }
							else { delete dnw; }
							recinsert<<<1,1,0,streams[4]>>>(tree->DNW,updatedBod);
						} else {
							delete dnw;
							Octant *dne = tree->octy->mDNE();
							if (contains(dne,updatedBod->position))
							{
								if (tree->DNE==NULL) { tree->DNE = new Bhtree(dne); }
								else { delete dne; }
								recinsert<<<1,1,0,streams[5]>>>(tree->DNE,updatedBod);
							} else {
								delete dne;
								Octant *dsw = tree->octy->mDSW();
								if (contains(dsw,updatedBod->position))
								{
									if (tree->DSW==NULL) { tree->DSW = new Bhtree(dsw); }
									else { delete dsw; }
									recinsert<<<1,1,0,streams[6]>>>(tree->DSW,updatedBod);
								} else {
									delete dsw;
									Octant *dse = tree->octy->mDSE();
									if (tree->DSE==NULL) { tree->DSE = new Bhtree(dse); }
									else { delete dse; }
									recinsert<<<1,1,0,streams[7]>>>(tree->DSE,updatedBod);
									}
								}
							}
						}
					}
				}
			}
	//	delete updatedBod;
		if (isExtern) {
			recinsert<<<1,1>>>(tree,insertBod);
		}
	}
}

__global__ void interactBodies(struct body* bods)
{
	// Sun interacts individually
	printf("\ncalculating force from star...");
	struct body *sun = &bods[0];
	for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
	{
		singleInteraction(sun, &bods[bIndex]);
	}

	//if (DEBUG_INFO) {std::cout << "\nBuilding Octree..." << std::flush;}
	printf("\nBuilding octree...");
	// Build tree
	vec3 *center = new struct vec3;
	center->x = 0;
	center->y = 0;
	center->z = 0.1374; /// Does this help?
	Octant *root = new Octant(center, 60*SYSTEM_SIZE);
	Gtree = new Bhtree(root);

	for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
	{
		if (contains(root,bods[bIndex].position))
		{
			recinsert<<<1,1>>>(Gtree,&bods[bIndex]);
			cudaError_t error = cudaGetLastError();
			if(error!=cudaSuccess)
				printf("\nCUDA error:%s",cudaGetErrorString(error));
		}
	}
	printf("\ncalculating interactions...");
	//if (DEBUG_INFO) {std::cout << "\nCalculating particle interactions..." << std::flush;}
	
	// loop through interactions
	//#pragma omp parallel for
	for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
	{
		if (contains(root,bods[bIndex].position))
		{
			interactInTree<<<1,1>>>(Gtree,&bods[bIndex]);
			cudaError_t error = cudaGetLastError();
                        if(error!=cudaSuccess)
                                printf("\nCUDA error:%s",cudaGetErrorString(error));
		}
	}
	
	// Destroy tree
//	delete Gtree;
	//
	printf("\nupdating particle positions...");
	//if (DEBUG_INFO) {std::cout << "\nUpdating particle positions..." << std::flush;}
	updateBodies(bods);
}

__global__ void interactInTree(Bhtree *tree, struct body* b)
{
	bool isExternal = tree->UNW==NULL && tree->UNE==NULL && tree->USW==NULL && tree->USE==NULL;
	isExternal = isExternal && tree->DNW==NULL && tree->DNE==NULL && tree->DSW==NULL && tree->DSE==NULL;
	const int num_streams=8;
	cudaStream_t streams[num_streams];
	for(int i=0; i<num_streams; i++)
		cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
	Octant *o = tree->octy;
	body *myb = tree->myBod;
	if(isExternal && myb!=b)
		singleInteraction(myb,b);
	else if(o->getLength()/magnitude(myb->position.x-b->position.x,
						  myb->position.y-b->position.y,
						  myb->position.z-b->position.z) < MAX_DISTANCE)
		singleInteraction(myb,b);
	else
	{
		if (tree->UNW!=NULL) interactInTree<<<1,1,0,streams[0]>>>(tree->UNW,b);
		if (tree->UNE!=NULL) interactInTree<<<1,1,0,streams[1]>>>(tree->UNE,b);
		if (tree->USW!=NULL) interactInTree<<<1,1,0,streams[2]>>>(tree->USW,b);
		if (tree->USE!=NULL) interactInTree<<<1,1,0,streams[3]>>>(tree->USE,b);
		if (tree->DNW!=NULL) interactInTree<<<1,1,0,streams[4]>>>(tree->DNW,b);
		if (tree->DNE!=NULL) interactInTree<<<1,1,0,streams[5]>>>(tree->DNE,b);
		if (tree->DSW!=NULL) interactInTree<<<1,1,0,streams[6]>>>(tree->DSW,b);
		if (tree->DSE!=NULL) interactInTree<<<1,1,0,streams[7]>>>(tree->DSE,b);
	}
}

__device__ inline bool contains(Octant *root, vec3 p)
{
	double length = root->getLength();
	vec3* mid = root->getMid();
	return p.x<=mid->x+length/2.0 && p.x>=mid->x-length/2.0 &&
			   p.y<=mid->y+length/2.0 && p.y>=mid->y-length/2.0 &&
			   p.z<=mid->z+length/2.0 && p.z>=mid->z-length/2.0;
}

__host__ __device__ double magnitude(float x, float y, float z)
{
	return sqrt(x*x+y*y+z*z);
}

__device__ void singleInteraction(struct body* a, struct body* b)
{
	vec3 posDiff;
	posDiff.x = (a->position.x-b->position.x)*TO_METERS;
	posDiff.y = (a->position.y-b->position.y)*TO_METERS;
	posDiff.z = (a->position.z-b->position.z)*TO_METERS;
	double dist = magnitude(posDiff);
	double F = TIME_STEP*(G*a->mass*b->mass) / ((dist*dist + SOFTENING*SOFTENING) * dist);

	a->accel.x -= F*posDiff.x/a->mass;
	a->accel.y -= F*posDiff.y/a->mass;
	a->accel.z -= F*posDiff.z/a->mass;
	b->accel.x += F*posDiff.x/b->mass;
	b->accel.y += F*posDiff.y/b->mass;
	b->accel.z += F*posDiff.z/b->mass;
}

__host__ __device__ double magnitude(vec3 v)
{
	return sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
}

__device__ void updateBodies(struct body* bods)
{
	double mAbove = 0.0;
	double mBelow = 0.0;
	for (int bIndex=0; bIndex<NUM_BODIES; bIndex++)
	{
		struct body *current = &bods[bIndex];
		if (DEBUG_INFO)
		{
			if (bIndex==0)
			{
			//	std::cout << "\nStar x accel: " << current->accel.x
			//			  << "  Star y accel: " << current->accel.y;
			} else if (current->position.y > 0.0)
			{
				mAbove += current->mass;
			} else {
				mBelow += current->mass;
			}
		}
		current->velocity.x += current->accel.x;
		current->velocity.y += current->accel.y;
		current->velocity.z += current->accel.z;
		current->accel.x = 0.0;
		current->accel.y = 0.0;
		current->accel.z = 0.0;
		current->position.x += TIME_STEP*current->velocity.x/TO_METERS;
		current->position.y += TIME_STEP*current->velocity.y/TO_METERS;
		current->position.z += TIME_STEP*current->velocity.z/TO_METERS;
	}
	if (DEBUG_INFO)
	{
		//std::cout << "\nMass below: " << mBelow << " Mass Above: "
		//		  << mAbove << " \nRatio: " << mBelow/mAbove;
	}
}

void createFrame(char* image, double* hdImage, struct body* b, int step)
{
	std::cout << "\nWriting frame " << step;
	if (DEBUG_INFO)	{std::cout << "\nClearing Pixels..." << std::flush;}
	renderClear(image, hdImage);
	if (DEBUG_INFO) {std::cout << "\nRendering Particles..." << std::flush;}
	renderBodies(b, hdImage);
	if (DEBUG_INFO) {std::cout << "\nWriting frame to file..." << std::flush;}
	writeRender(image, hdImage, step);
}

void renderClear(char* image, double* hdImage)
{
	for (int i=0; i<WIDTH*HEIGHT*3; i++)
	{
	//	char* current = image + i;
		image[i] = 0; //char(image[i]/1.2);
		hdImage[i] = 0.0;
	}
}

void renderBodies(struct body* b, double* hdImage)
{
	/// ORTHOGONAL PROJECTION
	for(int index=0; index<NUM_BODIES; index++)
	{
		struct body *current = &b[index];

		int x = toPixelSpace(current->position.x, WIDTH);
		int y = toPixelSpace(current->position.y, HEIGHT);

		if (x>DOT_SIZE && x<WIDTH-DOT_SIZE &&
			y>DOT_SIZE && y<HEIGHT-DOT_SIZE)
		{
			double vMag = magnitude(current->velocity);
			colorDot(current->position.x, current->position.y, vMag, hdImage);
		}
	}
}

double toPixelSpace(double p, int size)
{
	return (size/2.0)*(1.0+p/(SYSTEM_SIZE*RENDER_SCALE));
}

void colorDot(double x, double y, double vMag, double* hdImage)
{
	const double velocityMax = MAX_VEL_COLOR; //35000
	const double velocityMin = sqrt(0.8*(G*(SOLAR_MASS+EXTRA_MASS*SOLAR_MASS))/
			(SYSTEM_SIZE*TO_METERS)); //MIN_VEL_COLOR;
	const double vPortion = sqrt((vMag-velocityMin) / velocityMax);
	color c;
	c.r = clamp(4*(vPortion-0.333));
	c.g = clamp(fmin(4*vPortion,4.0*(1.0-vPortion)));
	c.b = clamp(4*(0.5-vPortion));
	for (int i=-DOT_SIZE/2; i<DOT_SIZE/2; i++)
	{
		for (int j=-DOT_SIZE/2; j<DOT_SIZE/2; j++)
		{
			double xP = floor(toPixelSpace(x, WIDTH));
			double yP = floor(toPixelSpace(y, HEIGHT));
			double cFactor = PARTICLE_BRIGHTNESS /
					(pow(exp(pow(PARTICLE_SHARPNESS*
					(xP+i-toPixelSpace(x, WIDTH)),2.0))
				       + exp(pow(PARTICLE_SHARPNESS*
					(yP+j-toPixelSpace(y, HEIGHT)),2.0)),/*1.25*/0.75)+1.0);
			colorAt(int(xP+i),int(yP+j),c, cFactor, hdImage);
		}
	}

}

void colorAt(int x, int y, struct color c, double f, double* hdImage)
{
	int pix = 3*(x+WIDTH*y);
	hdImage[pix+0] += c.r*f;//colorDepth(c.r, image[pix+0], f);
	hdImage[pix+1] += c.g*f;//colorDepth(c.g, image[pix+1], f);
	hdImage[pix+2] += c.b*f;//colorDepth(c.b, image[pix+2], f);
}

unsigned char colorDepth(unsigned char x, unsigned char p, double f)
{
	return fmax(fmin((x*f+p),255),0);
//	unsigned char t = fmax(fmin((x*f+p),255),0);
//	return 2*t-(t*t)/255;
}

double clamp(double x)
{
	return fmax(fmin(x,1.0),0.0);
}

void writeRender(char* data, double* hdImage, int step)
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

