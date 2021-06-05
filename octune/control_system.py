"""
BSD 3-Clause License

Copyright (c) 2021, Mohamed Abdelkader Zahana
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import control as cnt

class Control:
    """
    Simulates some control systems
    """
    def __init__(self):
        self._debug=True        # Print debug messages
        self._plant_tf=None     # Plant transfer function in Z domain
        self._cont_tf=None      # Controller transfer function in Z domain
        self._sys_tf=None       # Closed loop transfer function of the discrete system 
        self._dt=0.01           # Sampling time in seconds

        self._r=None            # System reference signal
        self._u=None            # Controller output signal
        self._y=None            # System output signal

    def buildPlantTF(self, num_coeff=None, den_coeff=None):
        """Builds plant transfer function given on numerator/denominator coeffs, and dt

        Uses self._dt

        Updates self._plant_tf

        @param num_coeff Array of numerator coeffs
        @param den_coeff Array of numerator coeffs
        """
        if (num_coeff is None):
            print("[ERROR] [buildPlantTF] Numerator num_coeff is None.")
            return False
        if (den_coeff is None):
            print("[ERROR] [buildPlantTF] Denominator coeffs den_coeff is None.")
            return False

        if (self._debug):
            print("[DEBUG] [buildPlantTF] Plant transfer funciton is created.")

        return True

    def buildControllerTF(self, num_coeff=None, den_coeff=None):
        """Builds controller transfer function given on numerator/denominator coeffs, and dt

        Uses self._dt

        Updates self._cont_tf

        @param num_coeff Array of numerator coeffs
        @param den_coeff Array of numerator coeffs
        """
        if (num_coeff is None):
            print("[ERROR] [buildControllerTF] Numerator num_coeff is None.")
            return False
        if (den_coeff is None):
            print("[ERROR] [buildControllerTF] Denominator coeffs den_coeff is None.")
            return False

        if (self._debug):
            print("[DEBUG] [buildControllerTF] Controller transfer funciton is created.")

        return True

    def buildSystemTF(self):
        """Builds closed loop system's transfer function given _plant/controller TFs

        Uses self._cont_tf, self._plant_tf

        Updates self._sys_tf
        """
        if (self.buildControllerTF()):
            print("[ERROR] [buildSystemTF] Error in building controller TF.")
            return False
        if (self.buildPlantTF()):
            print("[ERROR] [buildSystemTF] Error in building plant TF.")
            return False

        if (self._debug):
            print("[DEBUG] [buildSystemTF] Closed loop system transfer funciton is created.")

        return True

    def createStepInput(self, step=1.0, T=2.0):
        """Creates a step signal for a duration of T seconds
        Uses self._dt
        Updates self._r

        @param step Signal amplitude
        @param T Time length in seconds
        """
        pass

    def createPeriodicInput(self, mag=1.0, T=2.0):
        """Creates a sin wave for the reference signal
        Uses self._dt
        Updates self._r

        @param mag Signal amplitude
        @param T Time length in seconds
        """
        pass

def test():
    obj=Control()

if __name__== "__main__":
    test()