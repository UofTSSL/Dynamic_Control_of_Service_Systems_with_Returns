## Dynamic Control of Service Systems with Returns

Source code to reproduce the numerical results and plots in "Dynamic Control of Service Systems with Return: Application to Design of Post-Discharge Hospital Readmission Prevention Programs", by Timothy C. Y. Chan, Simon Y. Huang, and Vahid Sarhangian.

The code is in Python 3 and Jupyter Notebook and requires several packages as indicated at the top of the files. SimRNG, SimClasses, and SimFunctions are required to run the simulations and are provided in the folder Simulation Packages.  

## Code Description

The code is organized in several folders with the corresponding sections of the paper provided in brackets in the name of each folder:

1. Illustrative Examples: Contains the source code to compute the optimal policy under the three cost structures and reproduce the contour and stream plots Section 4.4.
2. Numerics - Stationary Arrivals: Contains the source code for simulation experiments and plots under stationary arrivals in Section 6.1, as well as the supplementary experiments in the E-Companion.
3. Numerics - Time-Varying Arrivals: Contains the source code for simulation experiments and plots under non-stationary arrivals in Sections 5 and 6.2, as well as the supplementary experiments in the E-Companion.
4. Case Study: Contains the source codes to reproduce the results and plots of the main case study.
5. Noisy Case Study: Contains the source code to reproduce the results and plots of the case study with noisy Content state provided in Section EC 4.4.
6. Optimal Fixed Return Probability: Contains the source code to compute the optimal fixed return probability for the stochastic system and reproduce the plots of Section EC 4.1.
7. Sample Path Convergence: Contains the source code for simulation experiments to compare the scaled sample paths with fluid trajectories and reporoduce the plots in Section EC 4.5.
8. Simulation Packages: Includes three files needed to run all simulation experiments. These need to placed in the same folder as the simulation source codes.

