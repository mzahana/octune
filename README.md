# OCTUNE: Online Control Tuning
OCTUNE is light-weight online optimal control tuning using backprobagation techniques with guranteed closed-loop stability.

# Installation
* Make sure that you have `pip` installed
    ```sh
    sudo apt install python3-pip -y
    ```
* Change directory to the folder of this package
* Install this package in `develop` mode, if you are planning to modify the code
    ```sh
    python3 setup.py develop --user
    ```
* Install this package in `install` mode for production
    ```sh
    python3 setup.py install --user
    ```
* To test the package, open a terminal and type
    ```sh
    python3 -m octune
    ```