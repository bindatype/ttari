#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>

/* Here we define all the constants relevant to the program as well as the calculations */
#define ncells 256
#define PI 3.14159265358979323846264338327
/* The accretion rate is taken to be 10^17 g/sec */

//Universal constants in CGS units.
#define c 2.99792458e10  //Speed of light in vacuum
#define h 6.6260755e-27  // Planck's Constant
#define G 6.67259e-8   // Universal Gravitational Constant
#define k 1.380658e-16  // Boltzmann Constant
#define sigma 5.67051e-5 // Stefan-Boltzmann Constant
#define M_0 1.9891e33  // The Solar Mass in grams.




// Parameters used in the calculation of Luminosity
#define T_in 1.0e5  // Temperature at the inner disk boundary
#define r_in 5.0e8 // Inner disk radius
#define r_out 2.5e10   //The outer disk radius is chosen to be 100 times the inner radius
#define M_star 0.5*M_0 // Mass of the Primary star
#define T_a 1.0e6    // The temperature of the disk atmosphere
#define f_r 1.0   // The function f(r) represents the inner boundary effect. If used it has to be declared as an array!
#define acc_rate 1.0e17  //This is the accretion rate per second
#define Arerad 1.0      // Disk absorption and reradiation efficiency is set at 100%
#define Adir 1.0        // Atmospheric (Direct) radiation is set at 100% but is currently not included in the total radiation
#define lambda_1 2.0e-5   //First wavelength window  (2000 Angstrom, UV range)
#define lambda_2 6.0e-5  //Second wavelegth window   (6000 Angstrom, Visible range)



// Parameters used by the code
#define Ndata 1024       // Number of data points in the light curve
#define Nwin 20          // Number of time windows over which averaging is performed to reduce statistical errors
#define bin 16          // Number of data points being binned (corresponds to bin times the freefall time)
#define mass_range 0.0  //This variable decides the range for the critical mass in each ring.
#define m_unit 1.0      // The basic unit of avalanching mass

#define time_freefall (1.0/ncells)*pow(r_out/(G*M_star),0.5)*(pow((r_out-r_in)*r_in,0.5)+r_out*acos(pow(r_in/r_out,0.5)))

