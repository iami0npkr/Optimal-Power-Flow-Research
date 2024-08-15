# Optimal-Power-Flow-Research
Optimal Power Flow Analysis for Power Systems

Nannuri Pranay Kumar Reddy , 21085052 , Btech, Part 4(Research Paper 1)
Macha Venkata Vasishta , 21085043 , Btech, Part 4(Research Paper 2)
Jarupula Anitha , 21085033 , Btech, Part 4(Research Paper 3)

-------------------------------------------------------------------------------------------------------------------------------
WEEK - 1

Research Paper 1: OPTIMUM POWER FLOW ANALYSIS BY NEWTON RAPHSON METHOD, A CASE STUDY

Description:

This research paper tells about optimal power flow(OPF) analysis in power system and how can this be done with newton raphson method. It tells about the objective of OPF and how to attain it using newton raphson method . And also some research is done to know how traditional load flow analysis differs from optimal power flow.

Literature:

Some key components we learnt though this paper are:

Load Flow Analysis
Optimal Power Flow Analysis
Newton Raphson Method
Objective functions for OPF

Summary:

Load Flow and OPF

Due to increasing load demands ,Power system grids get destabilise and may cause blackout etc. So frequently power systems should be analysed and should be modified according to the load demands. Load Flow (LF) and Optimal Power Flow (OPF) analyses are crucial for the efficient operation and analysing of power systems. Understanding the differences between these two methods is important.

Load Flow Analysis:
LF analysis is used to calculate the voltage magnitudes and angles at different buses, as well as the real and reactive power flows in the network under steady-state conditions. It helps in ensuring that the system operates within its limits under various load conditions.


Example:

Considering a simple power system with three buses (A, B, and C) connected by transmission lines. LF analysis can determine:
- The voltage at each bus (e.g., 1.0 p.u. at A, 0.98 p.u. at B, and 0.97 p.u. at C).
- The power flow on each line (e.g., 50 MW from A to B, 30 MW from B to C).

Optimal Power Flow Analysis:

OPF is same as LF but additionally it optimizes a specific objective function, such as minimizing generation cost, reducing power losses, or improving voltage stability. OPF takes into account additional constraints like generator limits, line capacities, and voltage limits(LF).

Example:

In the same three-bus system, OPF might aim to minimize the total generation cost while ensuring:
- The voltage at each bus remains within acceptable limits (e.g., 0.95 to 1.05 p.u.).
- The power flow on each line does not exceed its capacity (e.g., no more than 100 MW).

Key Differences Between LF and OPF:

- Objective: LF aims to determine the power system's steady-state conditions, whereas OPF aims to optimize system performance.

- Complexity: OPF is more complex due to the inclusion of optimization objectives and additional constraints.

- Applications: LF is used for routine analysis and planning, while OPF is used for operational optimization and advanced planning.

Newton-Raphson Method in LF and OPF:

The Newton-Raphson method is a iterative technique used for solving non-linear algebraic equations in both LF and OPF analysis. It is preferred due to its quadratic convergence and robustness.

Steps in Newton-Raphson Method:

1. Initial Guess: Start with an initial guess for the voltages and angles at each bus.
2. Jacobian Matrix Calculation: Compute the Jacobian matrix, which contains partial derivatives of the power flow equations with respect to the voltage magnitudes and angles.
3. Update Solution: Solving the linearized system of equations to update the voltage magnitudes and angles.
4. Iteration: Repeat the process until the changes in voltage magnitudes and angles are within acceptable limits.

Example:
For a bus with an initial voltage guess of 1.0 p.u., the Newton-Raphson method iteratively adjusts this value to converge to the true voltage value that satisfies the power flow equations.

Benefits of the Newton-Raphson Method:

- Accuracy:  Provides highly accurate results for LF and OPF problems.
- Convergence: Converges quickly, especially when the initial guess is close to the true solution.
- Robustness: Handles large and complex power systems effectively.

Steps in OPF Analysis:

1. Data Collection: we should gather data on the power system, including bus voltages, line impedances, and generator characteristics.

2. Model Formulation:  Develop mathematical models representing the power system and the optimization objectives.(For example we can simulate in Simulink).

3. Solution Techniques:  Use the Newton-Raphson(or any iterative method) method and other optimization algorithms to solve the OPF problem.

Benefits of OPF:
- Cost Efficiency: Reduces operational costs by optimizing the dispatch of generation units.
- Reliability: Enhances system reliability by ensuring operation within safe limits.
- Efficiency: Improves the overall efficiency of the power system by minimizing losses.

Conclusion

In conclusion, LF analysis is essential for understanding the current state of a power system, while OPF analysis provides a powerful tool for optimizing system performance. The Newton-Raphson method plays a critical role in solving both LF and OPF problems due to its accuracy and quickness . Implementing OPF with the Newton-Raphson method can lead to significant improvements in efficiency and reliability, making it a valuable technique for power system engineers.


-------------------------------------------------------------------------------------------------------------------------------




Research Paper 2:  Optical Power Flow Methods : A Comprehensive Survey


Description: In this research paper we are going to see just the methods used to solve Optimal Power Flow(OPF) Analysis.

Literature: Some key components we learnt in though this paper are:

Optimal Load Flow
Artificial Intelligence

Summary:
 
The optimal power flow solution methods are mainly classifies into two methods
Traditional Methods and Artificial Intelligence(AI) Methods
The Traditional Methods are further further subdivided, these are:
a)Linear Programming(LP)
In this method we use linear equations and inequalities to solve the  optimal power flow problems. The main advantage of using LP in OPF are its computational efficiency and its availability of robust solvers, making it effective for large-scale Power systems.However its only applicable to problems where the scenario could be approximated to linear models.

b)Gradient Method

As the name suggests, in this method we use the gradient(or derivative) of the objective function with respect to the decision variables such as generator outputs and voltage levels step by step i.e by iterations.This method is reliable, easy to implement, and converges for well-behaved functions, but they may take more time to converge or may get stuck in local minima(ex: Non-complex problems).

c)Newton-Raphson Method

Newton method is a second order method for unconstrained optimization based on the
application of a 2nd order Taylor series expansion about the current candidate solution. It is a iterative technique. In this method we calculate the Jacobian matrix and updating the variables using inverse jacobian matrix and function’s derivative. This method is known for its fast convergence, making it suitable for large-scale power systems.It can be computationally expensive due to matrix calculations.

d)Quadratic Programming

Quadratic Programming (QP) is a special form of non-linear programming whose objective
function is quadratic and constraints are linear. We solve optimal power flow problems using QP solvers.This method can handle more complex and realistic cost functions compared to linear programming.

e)NonLinear Programming

Nonlinear programming (NLP) in Optimal Power Flow (OPF) is an advanced optimization
technique used to determine the optimal operating conditions of a power system by minimizing a non-linear objective function subject to non-linear constraints. In this method we use iterative algorithms to find optimal solution.This method is ideal for realistic and detailed modeling of power systems.

f)Interior-Point Method

The objective of the Interior-Point method is to minimize an objective function while satisfying both equality and inequality constraints. In this method we solve optimal power flow using barrier functions to transform the constrained problem into a series of easier problems, then iteratively solves these problems by moving through the interior of the feasible region.It’s highly efficient for large scale problems and its faster in converging compared to other methods.

Now we are going to see the Artificial Intelligence Methods:

a)Genetic Algorithm 

The objective of this method is to find the optimal operating conditions of a power system by minimizing an objective function while meeting system constraints. We approach the optimal power flow problem by using a population of potential solutions that evolve over generations through selection, crossover, and mutation processes. This method is capable of handling complex, non-linear, and non-convex problems without requiring gradient information. Advantages this method can offer is that it can escape local optima, providing good solutions for highly complex and multi-modal problems. But these may be computationally expensive.

b)Particle Swarm Optimization

The objective of this method is to minimize an objective function while satisfying system constraints. We approach this problem by using a swarm of particles, where each particle represents a potential solution. Particles adjust their positions based on their own experience and the experience of neighboring particles.This method is effective for handling complex, non-linear, and multi-modal optimization problems.It’s simple to implement, can escape local optima, and does not require gradient information.

c)Artificial Neural Network

The objective of this method is to optimize power system operations by predicting optimal operating points that minimize an objective function while meeting constraints. We approach this method by training a neural network using historical data or simulation results to learn the complex relationships between system variables and optimal solutions.This method is capable of modeling highly non-linear and complex relationships in power systems.Tha advantages of this method,  once trained, ANNs provide fast and efficient predictions, are robust to noise, and can handle real-time optimization.The downside of this method is it requires a large amount of training data, can be computationally intensive to train, and may need careful tuning of network architecture and parameters.

d)Bee Colony Optimization

Bee Colony Optimization (BCO) in Optimal Power Flow (OPF) is a heuristic optimization technique inspired by the foraging behavior of honeybees.The objective of this method is to optimize power system operations by minimizing an objective function while satisfying system constraints.We approach this method by mimicking the behavior of bees searching for nectar, with "scout bees" exploring new areas and "worker bees" refining promising solutions.The advantages of this method are it is capable of escaping local optima and providing high-quality solutions through collective search and exploration strategies.

e)Differential Evolution

Differential Evolution (DE) in Optimal Power Flow (OPF) is a heuristic optimization algorithm inspired by evolutionary biology, used to solve complex optimization problems. The objective of this method is to minimize an objective function (e.g., generation cost, losses) while satisfying power system constraints. We approach this method by using a population of candidate solutions and iteratively evolves them through mutation, crossover, and selection processes to improve the solution.The advantages of this method are it’s simple to implement, robust, and capable of finding global optima or near-optimal solutions.

f)Grey Wolf Optimizer

The Grey Wolf Optimizer (GWO) in Optimal Power Flow (OPF) is a nature-inspired heuristic optimization technique modeled after the hunting behavior of grey wolves. The objective of this method is to optimize power system operations by minimizing an objective function while adhering to system constraints. We approach this method by mimicking the social hierarchy and hunting strategy of grey wolves, including leadership and collaborative searching, to explore and exploit the solution space. Advantages of this method are it’s simple to implement and capable of providing high-quality solutions by balancing exploration and exploitation.

g)Shuffled Frog-Leaping

Shuffled Frog Leaping Algorithm (SFLA) in Optimal Power Flow (OPF) is a population-based optimization technique inspired by the behavior of frogs leaping and sharing information in a swamp.The objective of this function is to optimize power system operations by minimizing an objective function (e.g., cost, losses) while satisfying system constraints.We approach this method by dividing the population of solutions into groups (memeplexes) that evolve independently, with frogs in each group "leaping" to explore new solutions and sharing information between groups to enhance overall performance.The advantages are as described-Balances exploration and exploitation effectively, and can find high-quality solutions by leveraging collective intelligence and collaboration.


Conclusion:

This paper reviews various optimization methods for solving Optimal Power Flow (OPF)
problems. Classical methods, despite significant advancements, have limitations such as the need for linearization and differentiability, potential to get stuck in local optima, poor
convergence, and inefficiency with large numbers of variables. In contrast, Artificial Intelligence (AI) methods offer greater versatility, effectively handling qualitative constraints and finding multiple optimal solutions in a single run. They are well-suited for multi-objective optimization and often excel in locating global optimum solutions.



-------------------------------------------------------------------------------------------------------------------------------






















Research Paper 3:  A Review on Optimal power flow problem and solution methodologies     

-------------------------------------------------------------------------------------------------------------------------------
Description:This research paper explores the essential role of Optimal Power Flow (OPF) in power systems.The paper provides a review of various optimization techniques used for solving OPF problems.The research covers the evolution of OPF methods, starting from the 1960s, and discusses how these techniques have been adapted to address modern issues such as cost minimization, power losses, and system stability.

Literature: Some key components we learnt in though this paper are:

Conventional methods for OPF
Artificial Intelligence method for OPF

Summary:

Gradient Method: 
This method is used in solving the Optimal Power Flow (OPF) problem by focusing on state and control variables. It simplifies the problem by reducing the number of variables through power flow equations. The method involves optimizing generation costs and reducing active losses using penalty functions, while also incorporating techniques like the Lagrange Multiplier for boundary verification.

Newton Method:
The Newton method is an effective solution algorithm known for its fast convergence when approaching a solution. It is particularly useful in power system applications where system voltages and generator outputs are near their nominal values. The method utilizes the Jacobian Matrix and Lagrangian Multipliers to optimize control variables.

 Linear Programming Method (LP) :
This method involves formulating the optimization problem with a linear objective function and linear constraints, using non-negative variables. The Simplex method is often employed to solve the LP problems by selecting and updating variables at different buses. However, LP methods may struggle with infeasible situations. Advanced techniques include using piece-wise differentiable penalty functions and mixed-integer LP to address specific constraints like contingency and reduce transmission losses. 
        This method requires linearization of both the objective function and constraints in each iteration to improve results, and it can handle discrete variables such as capacitors and shunt reactors.



Quadratic Programming (QP):
This method is a type of nonlinear programming where the objective function is quadratic and the constraints are linear. This method uses algorithms like Wolfe’s algorithm and Quasi-Newton techniques to solve the optimization problem. It is effective for various bus systems and can handle initial impractical starting points with fast convergence. The QP method is particularly useful for solving optimal power flow (OPF) problems, including those involving FACTS devices and large systems. It also addresses the sensitivity of solutions to different starting points and aims to achieve accurate results for both loss and cost minimization.

Interior Point (IP) Method:
This method is designed for solving large-scale linear programming (LP) problems efficiently. It operates within the interior of the feasible space, distinguishing it from methods like the Simplex method. The IP method has proven to be significantly faster in solving problems with many variables and constraints. Variations of the IP method, such as the Extended Quadratic Interior Point (EQIP) method and Interior Point Branch and Cut Method (IPBCM), have been developed for specific optimization challenges, including mixed-integer nonlinear problems. The IP method is applicable to both LP and quadratic programming (QP) problems and is often used for large-scale power system optimizations.

Particle Swarm Optimization (PSO) inspired by the social behaviors of animals to solve optimization problems.It is used in power systems for tasks like reactive power control and optimal power flow (OPF). PSO involves particles that adjust their positions based on their own and their neighbors' experiences. Enhanced PSO methods improve speed and accuracy by refining particle movement and handling constraints.

Artificial Bee Colony (ABC) Optimization inspired by  the behavior of honey bees to solve problems. It has been effectively used for optimal power flow (OPF) in power systems. The algorithm explores solutions efficiently, often converging faster than other methods. Variants include Chaotic ABC and hybrid methods combining ABC with Differential Evolution for better performance. ABC has shown success in various test systems, demonstrating its accuracy and effectiveness.

Conclusion:
This paper reviews methods for solving Optimal Power Flow (OPF) problems, highlighting that traditional methods often face limitations, such as getting stuck in local optima and struggling with complex constraints. In contrast, artificial intelligence techniques offer greater flexibility and better solutions for complex, multi-objective problems. As renewable energy integration increases, AI methods become crucial for effective decision-making and system optimization. This review provides a solid foundation for future research in OPF.

-------------------------------------------------------------------------------------------------------------------------------


REFERENCES:
Paper 1: Eltamaly, A. M., Sayed, Y., El-Sayed, A. H. M., & Elghaffar, A. N. A. (2018). Optimum power flow analysis by Newton raphson method, a case study. Ann. Fac. Eng. Hunedoara, 16(4), 51-58.
Paper 2: Khamees, A. K., Badra, N., & Abdelaziz, A. Y. (2016). Optimal power flow methods: A comprehensive survey. International Electrical Engineering Journal (IEEJ), 7(4), 2228-2239.
Paper 3: Maskar, M. B., Thorat, A. R., & Korachgaon, I. (2017, February). A review on optimal power flow problem and solution methodologies. In 2017 International Conference on Data Management, Analytics and Innovation (ICDMAI) (pp. 64-70). IEEE.

-------------------------------------------------------------------------------------------------------------------------------

WEEK - 2
Overview of Optimal Power flow
Economic load dispatch(ELD) calculates how much power is produced at each power generator ignoring the transmission line limits.
Optimal Power Flow(OPF) combines the load flow analysis and economic load dispatch.
Inshort ,the economic load dispatch is not considering the limits of real power losses due to which it may not provide the optimum economical scheduling of the power plants.
On the other hand, in optimal power flow, the losses are obtained first by performing load analysis in order to obtain the minimum losses so that cheapest overall generating cost is obtained.
 Problem Statement
Taken a 5-bus power system with generator at buses 1,2, and 3 .Bus 1 , with its voltage is set at 1.06 and angle is 0 degree p.u ,it is taken as slack bus


INPUT DATA
Input data is taken from a problem in the book “Power System Analysis by Hadi Sadat, Chapter 7.
Some input values like reactive power limits ,real power limits, generation power limits are assumed.and cost functions are the following:
C1=200 + 7.0*P1 + 0.008*P1*P1;
C2=180+ 6.3*P2 + 0.009*P2*P2;
C3=140 + 6.8*P3 + 0.007*P3*P3;
P1,P2,P3 are in MW 
10<=P1<=85
10<=P2<=80
10<=P3<=70
APPROACH
Here, we first calculated the  Ybus matrix , and then solved the power flow analysis by Newton Raphson Method. 
Active power Pi and reactive power Qi at bus i


1)Power Mismatch calculations

2)Jacobian Matrix

3)Jacobian Elements




4)Voltage and Magnitude Updates


From Newton Raphson we noted the total transmission losses and power generated by slack bus..
 Ploss between 2 buses.
Now, Solved the system using Economic Load Dispatch(ELD) , and found out the amount of power to be generated by slack bus. 
Now from the above values , we see the absolute difference between the power generated values from both methods , if the difference is less than 0.001 we will stop, else values are adjusted and the loops runs again.
CODE:
In the below code , it performs power flow analysis and optimizes power generation dispatch in an electrical grid. It calculates bus voltages and angles using the Newton-Raphson method, computes system losses, and adjusts generator outputs to minimize costs while meeting demand. The process iteratively refines the solution until the power system operates efficiently with minimal discrepancies(dpslack<0.001).


This below code shows us the Newton-Raphson method in Optimal Load Flow method which is used to find the most optimal distribution of power generation in power systems.It iteratively solves the nonlinear equations of power flow and adjusts the generator outputs to meet the load demands while considering the system constraints.








The below code shows the method to calculate the YBUS matrix.To calculate the Y-bus matrix, sum the admittances of all branches connected to each bus for diagonal elements and take the negative of the admittance for off-diagonal elements connecting two buses.

This program solves the coordination equation for optimizing power flow based on economic dispatch. To operate, it needs the total load demand (Pdt), the cost function matrix (cost), and the generator MW limits. If the MW limits are not specified, the powerflow will be carried out without considering these constraints. Additionally, if the base MVA and any loss coefficients (B, B0, and B00) are provided, the program will determine the optimal dispatch while incorporating system losses.





The below code contains how the generation cost is calculated.We do this by firstly obtaining the power generation values for each generator and then using the cost function, we calculate the generated cost for each generator, the sum of all cost values will give us the total cost generated.

This code prints us the values that are calculated in steady state when doing Newton - Raphson method.


This program  calculates the B-coefficients of the loss formula as the function of real power  generation.                       PL = P*B*P'+B0*P'+B00.
It requires the power flow solution..




The below snippet will be the results that were calculated after doing power flow analysis and economic load dispatch. Here we obtained the results for each iteration during the Newton - Raphson method, the total cost using this method, and the total cost after doing the economic load dispatch.







…
…

CONCLUSION
Cost calculated after Newton - Raphson Power flow method is - 1,30,659.07 Rs/hr
Cost calculated after Optimal Power Flow(loadflow+ELD) - 1,27,757.17 Rs/hr
So for 1hr -  2901.9 Rs are saved
For 1 day - 69,645.6 Rs are saved
For 1 month - 20,89,368 Rs are saved
For 1 year - 2,50,72,416 Rs are saved.
This is for a small grid (5 Bus system) this much of amount is saved, In larger scale power grids much more money is saved.
The optimal Power flow solution is calculated successfully.




















