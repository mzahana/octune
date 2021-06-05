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
import scipy as sp
import scipy.signal

class BackProbOptimizer:
    """Implements backprobagation techniques to optimize a linear controller for an unkown system, given system & controller output data.
    Reference: TODO include a reference paper (possibly on arxiv)
    """
    def __init__(self):
        self._debug=True    # Print debug messages

        # System signals
        self._r=None        # System's reference signal
        self._y=None        # System output
        self._u=None        # Controller output

        # Controller paramaers, to be optimized
        self._a=None        # Denomenator coeffs. _a[0]=1
        self._b=None        # Numerator coeffs
        self._new_a=None# Updated a (denominator) coeffs after performing backward probagation
        self._new_b=None# Updated b (numerator) coeffs after performing backward probagation

        # Partial derivatives
        self._dL_dy=None    # Partial derivatives w.r.t to system output y
        self._dL_du=None    # Partial derivatives w.r.t to controller output u
        self._dL_da=None    # Partial derivatives w.r.t to controller denomentator's coeffs, a
        self._dL_db=None    # Partial derivatives w.r.t to controller numerator's coeffs, b


        # TODO ADAM optimizer variables
        self._use_adam=True # True: Use  ADAM algorithm (recommended); False: Use regular gradient descent aglrithm

    def backward(self):
        """Performs backward probagation steps.
        Computes partial derivatives of the performance function L, w.r.t controller parameters, controller output (u), system output (y)

        Uses _y, _u, _a, _b, and the ADAMS variabels.

        Updates _dL_da, _dL_db, _dL_dy, _dL_du

        @return True: if sccessful, False if not
        """

        if (self._debug):
            print("[DEBUG] [backward] Executing backward()...")

        # Sanity checks
        if (self._r is None):
            print("[ERROR] [backward] System reference signal r is None.")
            return False
        if (self._u is None):
            print("[ERROR] [backward] controller output signal u is None.")
            return False
        if (self._y is None):
            print("[ERROR] [backward] system output signal y is None.")
            return False
        if (len(self._u) != len(self._y)):
            print("[ERROR] [backward] Lengths of controller output signal u and system output signal y are not equal.")
            print("[ERROR] [backward] len(u)={}, len(y)={}".format(len(self._u), len(self._y)))
            return False
        if (len(self._r) != len(self._y)):
            print("[ERROR] [backward] Lengths of system reference signal r and system output signal y are not equal.")
            print("[ERROR] [backward] len(u)={}, len(y)={}".format(len(self._r), len(self._y)))
            return False

        if (self._a is None):
            print("[ERROR] [backward] a coeffs are None.")
            return False
        if (self._b is None):
            print("[ERROR] [backward] b coeffs are None.")
            return False

        # TODO implement the backward probagation steps

        # Compute error

        return True


    def update(self, iter=None):
        """Updates controller parameters _a & _b using Gradient descent or ADAM algorithms.
        Uses _dL_da, dL_db, and the learning rates; and ADAM variables if _use_adam=True

        Updates _new_a, _new_b
        
        @param iter iteration number
        @return True if successful, False if not
        """
        if (self._debug):
            print("[DEBUG] [update] Executing update()...")
            
        # sanity checks
        if (iter is None):
            print("[ERROR] [update] iteration number is None")
            return False
        if (self._dL_da is None):
            print("[ERROR] [update] partial derivatives dL_da is None.")
            return False

        if (self._debug):
            print("[DEBUG] [update] ************* Iteration number {} *************".format(iter))

        if ( self.backward() ):
        
            if (self._use_adam):
                # Use ADAM algorithm
                pass
            else:
                # Use regular gradient descent algorithm
                pass

    def setU(self, u=None):
        """This is a setter function for class variable _u, controller output.

        @param u array of controller output u. It should have similar length as system output y

        @return True if successful, False if not
        """
        if (u is None):
            print("[ERROR] [setU] u is None.")
            return False
        self._u = u

        if (self._debug):
            print("[DEBUG] [setU] u is set and of length {}".format(len(self._u)))

        return True

    def setY(self, y=None):
        """This is a setter function for class variable _y, system output.

        @param y array of controller output y. It should have similar length as system output u

        @return True if successful, False if not
        """
        if (y is None):
            print("[ERROR] [setY] y is None.")
            return False
        self._y = y

        if (self._debug):
            print("[DEBUG] [setY] y is set and of length {}".format(len(self._y)))
            
        return True

    def setContCoeffs(self, a=None, b=None):
        """Setter function for controller coeffs a,b

        @return True if successful, False if not
        """
        if (a is None):
            print("[ERROR] [setContCoeffs] a is None.")
            return False
        if (b is None):
            print("[ERROR] [setContCoeffs] b is None.")
            return False
        if (a[0] != 1):
            print("[ERROR] [setContCoeffs] a[0] is not 1.")
            return False

        self._a = a
        self._b = b

    def getNewContCoeff(self):
        """Returns _new_a, _new_b
        """
        return self._new_a, self._new_b

    def getDerivatives(self):
        """Return _dL_dy, _dL_du, _dL_da, _dL_db
        """
        return self._dL_dy, self._dL_du, self._dL_da, self._dL_db

def test():
    obj = BackProbOptimizer()

if __name__ == "__main__":
    test()