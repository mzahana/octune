# OCTUNE: Online Control Tuning
OCTUNE is light-weight online optimal control tuning using backprobagation techniques with guranteed closed-loop stability.

**NOTE** The current working version is tested with Python2 as it was needed for working with ROS Melodic

# Installation
$ Clone this package
    ```bash
    git clone https://github.com/mzahana/octune.git
    ```
* Change directory to the folder of this package
* Install this package in `develop` mode, if you are planning to modify the code
    ```sh
    python setup.py develop --user
    ```
* Install this package in `install` mode for production
    ```sh
    python setup.py install --user
    ```
# Testing
* To test the package, open a terminal and type
    ```sh
    python -m octune
    ```
* `octune/__main__.py` has an example on how to tune a controller
# Citation
If you use this work in your research, please cite the following reference.
```

@Article{s22239240,
AUTHOR = {Abdelkader, Mohamed and Mabrok, Mohamed and Koubaa, Anis},
TITLE = {OCTUNE: Optimal Control Tuning Using Real-Time Data with Algorithm and Experimental Results},
JOURNAL = {Sensors},
VOLUME = {22},
YEAR = {2022},
NUMBER = {23},
ARTICLE-NUMBER = {9240},
URL = {https://www.mdpi.com/1424-8220/22/23/9240},
ISSN = {1424-8220},
ABSTRACT = {Autonomous robots require control tuning to optimize their performance, such as optimal trajectory tracking. Controllers, such as the Proportional&ndash;Integral&ndash;Derivative (PID) controller, which are commonly used in robots, are usually tuned by a cumbersome manual process or offline data-driven methods. Both approaches must be repeated if the system configuration changes or becomes exposed to new environmental conditions. In this work, we propose a novel algorithm that can perform online optimal control tuning (OCTUNE) of a discrete linear time-invariant (LTI) controller in a classical feedback system without the knowledge of the plant dynamics. The OCTUNE algorithm uses the backpropagation optimization technique to optimize the controller parameters. Furthermore, convergence guarantees are derived using the Lyapunov stability theory to ensure stable iterative tuning using real-time data. We validate the algorithm in realistic simulations of a quadcopter model with PID controllers using the known Gazebo simulator and a real quadcopter platform. Simulations and actual experiment results show that OCTUNE can be effectively used to automatically tune the UAV PID controllers in real-time, with guaranteed convergence. Finally, we provide an open-source implementation of the OCTUNE algorithm, which can be adapted for different applications.},
DOI = {10.3390/s22239240}
}
```
