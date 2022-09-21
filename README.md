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
