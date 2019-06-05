#Vertical-Integration
This project is built in a modular way.
Generally speaking it can be split up into the following parts:
1. PDEs, their implementation and solutions. (&rarr; `cases`)
2. operators, to implement the PDEs. (&rarr; `operators`)
3. (time) integrators, to solve the PDEs numerically. (&rarr; `integrators`)
4. debugging and testing-tools, to verify the other three parts (&rarr; `debug_tools`, but parts are contained in the folders of the files that are to be tested).

##File Descriptions:
* `utils.py` contains classes that are needed by the entire project.
    * `State`: a class which stores state variables and their axis labelling.
    * `Integrator`: parent class for all integrators implemented in this project.
    * `Solution`: parent class for all solutions (analytical and numerical) in this project.
* `starting_conditions.py` contains a number of functions, which take in numpy arrays and apply a function to them.
The intended use of these functions is to set the starting condition for a PDE, and they are also used to construct solutions for some PDEs.
* `playground.py` This file is supposed to be used for prototyping. Currently it contains a working example for running a simulation using the exponential Integrator.

##Directory Descriptions:
* `cases` contains a folder structure of different implemented example-PDEs (e.g. Wave equation).
Each leaf-folder in this folder-tree contains the following three to four files: `derivative`, `run`, `test` (`solution`).
For descriptions refer the the README.md file in the cases-folder.
Also contains a framework for running said cases.
* `data` this folder is intended to be used to store numerical solutions for a given case. **You have to set the variable `data_path` in the file `utils.py` to the path to this folder**.
* `debug_tools` contains classes and methods useful for debugging, such as an `ErrorTracker`, and a visualization framework which supports displaying some of the common objects which contain data in this project (e.g. `State` and `ErrorTracker`).
* `integrators` contains classes for different time integrators, such as Runge-Kutta methods up to order 4, and a prototype for an exponential integrator.
Also contains unit-tests for Explicit Euler, Heun and RK4.
* `operators` contains differential operators (derivative, laplacian [second derivative], and averaging) with different error orders.
Also contains unit-tests for all operators.
* `thesis` contains .tex-files for the BA-Thesis.