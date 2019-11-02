# ViscousWaveApproximation
A nonlinear regression approach for my scientific problem. The approximation of the fluid velocity field which is obtained from numerical solution of the Navier-Stokes eq.

Approximation of viscous waves v1.0.
Developed by Artem Pavlovskii
October 2019.

This is free software that is distributed under the GNU General Public Licence. You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation.

We approximate data from numerical solution by dimensionless formula
v = Re((exp((j-1)x) + A*exp(-jC)(1-exp((j-1)x))*exp(-Bx-jt)) 
using the least squares and gradient descent methods. We denote by j the imaginary unit, by x the dimensionless coordinate and by t dimensionless time. The A, B and C are unknown coefficients which we will try to predict.
Please, see my article https://link.springer.com/article/10.1134/S1063784216070185 for the details.


The data files are obtained from the Comsol Multiphisics software. They must have only two columns 'x' and 'v' and 8 lines of the header. The value in the 'x' column should start from 0. The sequence in the 'x' column may repeat several times starting from 0 every time. The data must be obtained in the time interval from nT to (n+1)T or from nT+T/2 to (n+1)T+T/2 where T is the period of wall oscillations, n is a natural number.

The program is interactive. You must specify the maximum number of iterations, the coefficient's initial values and its limits. Note, that the result mostly depends on the inital values.
