#include "ttari.h"
/*
 *
 * Main Exec Time: 22.6 [sec]
 * Kernel Exec Time: 0.039648 [msec]
 * Iterations: 50481 [msec]
 * maclach@gpudev:~/sand_pile$ ./ttari
 * 2.973751e+00Main Exec Time: 49.43 [sec]
 * Kernel Exec Time: 0.380672 [msec]
 * Iterations: 50481 [msec]
 *
 *
 *
 * Main Exec Time: 33.59 [sec]
 * Kernel Exec Time: 0.275616 [msec]
 * Iterations: 50481 [msec]
 *
 * Main Exec Time: 20.44 [sec]
 * Kernel Exec Time: 0.0136 [msec]
 * Iterations: 50481 [msec]
 *
 */
using namespace std;

// i is the radial dimension


__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
	int id = threadIdx.x;
	curand_init ( seed, id, 0, &state[id] );
}


__global__ void visc_inflow(curandState* globalState, double *M,
		const double d_mv) {
	int count = 0;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curandState localState = globalState[idx];
	int ran = int (ncells*curand_uniform( &localState )) % (ncells);
	printf("Ran = %d %d \n",ran,idx);
	globalState[idx] = localState;

	if (idx <  ncells) {
		M[idx * ran] -= d_mv;
		if (idx < ncells - 1)
/*
			 This creates a race condition but it doesn't matter, most likely.
			 The condition is one of decrement/increment. Because addition is
			 commutative it doen't matter which executes first.
*/
			M[(idx + 1) * idx + ran] += d_mv;
	}
}


__global__ void do_avalanche(double *M, double *M_next, double *m, double *m_crit) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ((idx < ncells) && (blockDim.x == ncells)) {
		int i = blockIdx.x;
		int j = threadIdx.x;
		if (M[i * ncells + j] > m_crit[i]) {
			M[i * ncells + j] -= 3 * m[i];
			if (i < ncells - 1) {
				M_next[(i + 1) * ncells + j] += m[i];
				M_next[(i + 1) * ncells + (j + 1) % ncells] += m[i];
				M_next[(i + 1) * ncells + (j - 1 + ncells) % ncells] += m[i];
			}
			//	aval_count += 1;
		}

/*					for (j = 0; j < ncells; j++) {
						if (M[i * ncells + j] > m_crit[i]) {
							M[i * ncells + j] -= 3 * m[i];
							if (i < ncells - 1) {
								M_next[(i + 1) * ncells + j] += m[i];
								M_next[(i + 1) * ncells + (j + 1) % ncells] += m[i];
								M_next[(i + 1) * ncells + (j - 1 + ncells) % ncells] +=
										m[i];
							}
							// This is just
							aval_count += 1;

						}
						// This sum can be done later on CPU, rather not do this in device kernel. ./GAM
			//			sum += M[i * ncells + j];

					}*/

	}
}

__global__ void update_mass_arrays(double *M, double *M_next){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ((idx < ncells) && (blockDim.x == ncells)) {
		M[idx] += M_next[idx];
	}
}

__global__ void update_m_crit(double *m, double *m_crit){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ((idx < ncells) && (blockDim.x == 1)) {
		// Here we set the critical mass to vary over a given range (mass_range) starting from the initial (m_crit[0])
		m_crit[idx] = m_crit[0] + idx * mass_range * m_unit / (ncells - 1);
		// The condition below allows for the mass flow during an avalanche in each ring to equal the variation in critical mass
		m[idx] = m_crit[idx] * m[0] / m_crit[0];
	}
}


__global__ void init_2d_array(double *M, double value){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ((idx < ncells) && (blockDim.x == ncells)) {
		M[idx] = value;
	}
}
/* The main program begins here */
int main() {

	float milliseconds = 0;

	int i, j, points, ran_j, aval_count, r_num, Npts = Ndata * Nwin;
	int nsoc = 30000;   // where SOC is achieved
	int niter = nsoc + Ndata * Nwin + 1; //nsoc + Nwin times Ndata, 1024 data points averaged over 20 windows
	int temp1, knum, sum;
	double t_free, t, lum_tot;
	double mv = 0.1 * m_unit;
	double *m, *m_crit; // 1D ncells
	double *M_next, *M;  // 2D ncells by ncless
	double *d_m, *d_m_crit;
	double *d_M_next, *d_M;
	double m_visc, radius, temp2, r_width, f_1, f_2;
	double delta_m;
	double delta_A;
	double f, B_f, delta_f;
	double T, T_obs, delta_r[ncells], r[ncells], delta_E, delta_L_a,
	delta_L_a_f, delta_L_ring_tot, delta_L_ring_f, lum_tot_f[Npts],
	lum_tot_f_ave[Ndata];

	/* End of variable declaration block */

	/* Here we declare all the data files in which various data will be stored */
	/* Declaration of file pointers */
	FILE *rad1, *lc, *lum_ring, *lum_disk_all_points, *bin_data, *stdev,
	*ring50, *ring100, *ring150;
	//        FILE *rad2;
	/* End of the file pointer declaration block */

	lc = fopen("lc.dat", "w");
	lum_ring = fopen("lum_ring.txt", "w");
	lum_disk_all_points = fopen("lum_disk_all_points.txt", "w");
	bin_data = fopen("lum_disk_f_bin.txt", "w");
	stdev = fopen("std_deviation.txt", "w");
	ring50 = fopen("50th_ring.txt", "w");
	ring100 = fopen("100th_ring.txt", "w");
	ring150 = fopen("150th_ring.txt", "w");
	/* End of data files declaration block */


	/* CUDA TIMING SETUP */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (NULL == (M = (double *) malloc(ncells * ncells * sizeof(double))))
		exit(1);
	if (NULL == (M_next = (double *) malloc(ncells * ncells * sizeof(double))))
		exit(1);
	if (NULL == (m_crit = (double *) malloc(ncells * sizeof(double))))
		exit(1);
	if (NULL == (m = (double *) malloc(ncells * sizeof(double))))
		exit(1);

	t_free = time_freefall;
	m_visc = 0.1 * acc_rate * t_free;
	m_crit[0] = 3.0 * m_unit;
	m[0] = 1.0 * m_unit;
	knum = Ndata / bin;


	printf("%e\n", t_free);

	/* In this block we obtain the ring radii and areas generated by the radius_brute program */
	rad1 = fopen("radius_brute.txt", "r");
	// rad2=fopen("radius_ttari.txt","w");
	for (r_num = 0; r_num < ncells; r_num++) {
		fscanf(rad1, "%d %lf %lf %lf", &temp1, &radius, &temp2, &r_width);
		r[r_num] = radius;

		delta_r[r_num] = r_width;
		// fprintf(rad2,"%d %lf %lf\n", r_num, r[r_num], delta_r[r_num]);
	}
	fclose(rad1);
	// fclose(rad2);
	/* End of radii and area acquisition block */


	update_m_crit<<<1,ncells>>>(m,m_crit);

	/* End of avalanche conditions block */






	long startTime = clock();
	/*The data structures we want to move to device are:	 M	 M_next	 m_crit	 m	 */
	if (cudaSuccess != cudaMalloc((void**) &d_M, ncells * ncells * sizeof(double))){printf ("cudaMalloc Failed!\n");}
	cudaMemcpy(d_M, M, ncells * ncells * sizeof(double),
			cudaMemcpyHostToDevice);
	if ( cudaSuccess != cudaMalloc((void**) &d_M_next, ncells * ncells * sizeof(double))){printf ("cudaMalloc Failed!\n");}
	cudaMemcpy(d_M_next, M_next, ncells * ncells * sizeof(double),
			cudaMemcpyHostToDevice);

	if ( cudaSuccess != cudaMalloc((void**) &d_m_crit, ncells * sizeof(double))){printf ("cudaMalloc Failed!\n");}
	cudaMemcpy(d_m_crit, m_crit, ncells * sizeof(double),
			cudaMemcpyHostToDevice);
	if ( cudaSuccess != cudaMalloc((void**) &d_m, ncells * sizeof(double))){printf ("cudaMalloc Failed!\n");}
	cudaMemcpy(d_m, m, ncells * sizeof(double), cudaMemcpyHostToDevice);


	curandState* devStates;
	if ( cudaSuccess != cudaMalloc((void**) &devStates, ncells * sizeof(curandState))){printf ("cudaMalloc Failed!\n");}


	setup_kernel <<< ncells, ncells >>> ( devStates, time(NULL) );
	init_2d_array<<<ncells,ncells>>>(d_M,2.9);
	/* Clear all the cells to begin the simulation */


	//	/* This is the beginning of the simulation */  // BEGINNING OF THE ITER LOOP
	for (int iter = 0; iter < niter; iter++) { //nsoc + Ndata * Nwin + 1;
		//nsoc + Nwin times Ndata, 1024 data points averaged over 20 windows

		visc_inflow<<<1, ncells>>>(devStates, d_M, mv);

		/*This block performs accretion and the avalanche mass flow */
		sum = 0.;
		lum_tot = 0.;

		// The next mass value matrix is initialized to zero for each iteration
		init_2d_array<<<ncells,ncells>>>(d_M_next,0.);

		//?		ran_j = rand() % ncells;
		//?		M[ran_j] += 1.0 * m[0]; /*Randomly insert a mass (time of free fall multiplied by accretion rate) into the outermost ring*/

		cudaEventRecord(start);
		do_avalanche<<<ncells,ncells>>>(d_M, d_M_next, d_m, d_m_crit);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		for (i = 0; i < ncells; i++) {

			aval_count = 0;

			/* This is the block in which all the physics is applied i.e. all the calculations are made for radiation from a single ring */
			//			T = T_in * pow((r[i] / r_in), -0.75) * f_r;
			//			delta_m = 3 * acc_rate * m[i];
			//			delta_E = 0.5 * G * M_star * delta_m * delta_r[i]
			//					/ (r[i] * (r[i] - delta_r[i]) * t_free);
			//			delta_A = 2 * PI
			//					* (2 * r[i] * delta_r[i] - delta_r[i] * delta_r[i]);
			//			B_f = 2 * h * f * f * f / (c * c * (exp(h * c / (k * T)) - 1));
			//			delta_L_a = Adir * delta_E;
			//			delta_L_a_f = 4.9e-11 * delta_L_a * exp(-h * f / (k * T_a))
			//					+ delta_A * aval_count * B_f;
			//			delta_L_ring_tot = sigma * delta_A * pow(T, 4) * f_r
			//					+ Arerad * aval_count * delta_E;   //+Arerad*m_visc*delta_E;
			//					//lum_tot += delta_L_ring_tot;
			//			T_obs = pow(delta_L_ring_tot / (sigma * delta_A), 0.25);
			//			f = c / lambda_1;
			//			f_1 = c / (lambda_1 - 0.1e-5);
			//			f_2 = c / (lambda_1 + 0.1e-5);
			//			delta_f = f_1 - f_2;
			//			delta_L_ring_f = delta_A * 4.0 * PI * 2 * h * f * f * f * delta_f
			//					/ (c * c * exp(h * f / (k * T_obs)));
			//			/* End of calculations block */

			lum_tot += delta_L_ring_f;
		}
		/* End of  each iteration (corresponds to one free fall time) */


		/* Here the Disk is initialized to cell mass values for the next iteration */
		update_mass_arrays<<<ncells,ncells>>>(d_M,d_M_next);

		/* End of cell mass reinitialization block */

		/* The standard deviation of the mass distribution is calculated in the following block for every 100th iteration */
		//		if (iter % 100 == 0) {
		//			double mean = sum / (ncells * ncells);
		//			double sum_of_sq = 0.;
		//			for (i = 0; i < ncells - 1; i++) {
		//				for (j = 0; j < ncells; j++) {
		//					double temp = M[i * ncells + j] - mean;
		//					sum_of_sq += temp * temp;
		//				}
		//			}
		//			double std_dev = sqrt(sum_of_sq) / ncells;
		//			fprintf(stdev, "%d %e %e\n", iter, mean, std_dev);
		//		}
		/* End of the standard deviation block */

		/*This block looks at the masses in the 50th, 100th, 150th rings at the end of the simulation */
		/*		if (iter == niter - 1) {
		 for (i = 0; i < ncells - 1; i++) {
		 if (i == 49) {
		 for (j = 0; j < ncells; j++) {
		 fprintf(ring50, "%d %e \n", j, M[i * ncells + j]);
		 }
		 }
		 if (i == 99) {
		 for (j = 0; j < ncells; j++) {
		 fprintf(ring100, "%d %e \n", j, M[i * ncells + j]);
		 }
		 }
		 if (i == 149) {
		 for (j = 0; j < ncells; j++) {
		 fprintf(ring150, "%d %e \n", j, M[i * ncells + j]);
		 }
		 }

		 }
		 }*/
		/* End of the ring mass distribution block */

		/* In this block we record the data after SOC has been achieved */
		//		if (iter > nsoc) {
		//			points = (iter - nsoc) - 1;
		////			cout << "Points: " << points << endl;
		//			lum_tot_f[points] = lum_tot;
		//			fprintf(lum_disk_all_points, "%e %e\n", iter * t_free, lum_tot);
		//		}
	}                                                    // END OF THE ITER LOOP
	/* The simulation ends here */
	long finishTime = clock();

	//	/* This block averages the data for Ndata points over Nwin identical time windows to minimize statistical errors*/
	//	for (i = 0; i < Ndata; i++) {
	//		lum_tot_f_ave[i] = 0.0; // Initializes the time averaged data array to zero
	//	}
	//
	//	for (i = 0; i < Ndata; i++) {
	//		for (j = 0; j < Nwin; j++)
	//			lum_tot_f_ave[i] += (1.0 / Nwin) * lum_tot_f[j * Ndata + i];
	//		t += t_free;
	//		fprintf(lc, "%e %e\n", t, lum_tot_f_ave[i]);
	//	}
	/* End of time data averaging block */

	/* This block does the binning of the data */
	//	for(int pts = 0; pts < knum; pts ++)
	//	{
	//		for(i = bin_pts; i < bin_pts + bin; i++)
	//			lum_disk_f_bin[pts] += lum_tot_f_ave[i];
	//		t_bin += t_free*bin;
	//		fprintf(bin_data, "%e %e\n", t_bin, lum_disk_f_bin[pts]);
	//		bin_pts += bin;
	//	}
	/*  End of the time-binning block */

	cout << "Main Exec Time: "
			<< (finishTime - startTime) / (double) CLOCKS_PER_SEC << " [sec]"
			<< endl;
	cout << "Kernel Exec Time: " << milliseconds << " [msec]" << endl;
	cout << "Iterations: " << niter << "    NCells: " << ncells<< endl;
	free(m);
	free(M);
	free(M_next);
	free(m_crit);
	/* Here we close all the data files */
	fclose(lc);
	fclose(lum_ring);
	fclose(lum_disk_all_points);
	fclose(bin_data);
	fclose(stdev);
	fclose(ring50);
	fclose(ring100);
	fclose(ring150);
	return 0;
}
/* This is the end of the main program */

