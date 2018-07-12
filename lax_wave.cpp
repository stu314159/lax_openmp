#include "omp.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <chrono>

// spatial initialization 
void initialize_spatial_array(float * X, const float x_l, const float dx,
		const int N)
{
	X[0] = x_l;
	for(int i = 1; i < N; i++)
    {
		X[i] = X[i-1]+dx;
    }
}

void initialize_solution_array(float * F,const float * X, const int N)
{
    for(int i = 0; i<N; i++)
    {
    	F[i] = exp(-X[i]*X[i]);
    }
}

void write_output(const float * X, const float * F, const int N)
{
	std::ofstream out_file;
	out_file.open("lax_wave_out.bin", std::ios::out | std::ios::binary);
	for(int i = 0; i < N; i++)
	{
		out_file << X[i] << "  " << F[i] << std::endl;
	}
	out_file.close();

}


int main(int argc, char* argv[])
{
// user will input N and Num_ts
	const int N = atoi(argv[1]);
	const int Num_ts = atoi(argv[2]);
        const int WRITE_OUT = atoi(argv[3]); 
	// basic problem parameters
	const float x_left = -10; //m, left edge of domain
	const float x_right = 10; //m, right edge of domain
	const float dx = (x_right - x_left)/(N-1); // grid point spacing
	const float u = 1.; //m/s, wave speed
	const float dt = 0.6*dx/u; //s, time step size
	const float nu = u*dx/dt; //courant number

	float * X = new float[N]; // spatial array
	initialize_spatial_array(X,x_left,dx,N);

	float * F_even = new float[N]; // solution array
	float * F_odd = new float[N];
	float * F; float * F_new;
	initialize_solution_array(F_even,X,N);

        auto start = std::chrono::high_resolution_clock::now();



	// main time stepping loop
//#pragma omp target map(alloc:F_even[:N]) map(alloc:F_odd[:N])
#pragma omp target data map(tofrom: F_even[:N], F_odd[:N])
{
	for(int ts = 0; ts<Num_ts; ts++)
	{
		if ((ts%1000)==0)
		{
			std::cout << "Executing time step " << ts << std::endl;
		}

		if ((ts%2)==0)
		{
			F = F_even; F_new = F_odd;
		} else {
			F = F_odd; F_new = F_even;
		}
#pragma omp target teams   
#pragma omp distribute parallel for schedule(static,1)
		for(int i = 0; i < N; i++)
		{
			int x_m; int x_p;
			x_m = i-1; x_p = i+1;

			// apply periodic boundaries
			if (x_m < 0)
			{
				x_m = N-1;
			}

			if (x_p == N)
			{
				x_p = 0;
			}
			// apply the Lax Method
			F_new[i] = 0.5*(F[x_p] + F[x_m]) - (u*dt/(2.*dx))*(F[x_p] - F[x_m]);
		}

	}
}
	auto finish = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";

       if (WRITE_OUT == 1)
       {
	// time stepping complete, write the output
	    std::cout << "Writing output." << " \n";
	    write_output(X,F,N);
        }

	// clean up environment
	delete [] X;
	delete [] F_even;
	delete [] F_odd;

    return 0;
}
