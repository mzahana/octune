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
from numpy.lib.function_base import select
import scipy as sp
# import scipy.signal

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

        # Optimization objective function
        self._objective=None
        # Error array between reference signal and systen output
        self._error=None


        # Optimization learning rate
        self._alpha=0.001

        # TODO ADAM optimizer variables
        self._use_adam=True # True: Use  ADAM algorithm (recommended); False: Use regular gradient descent aglrithm
        self._beta1=0.9
        self._beta2=0.999
        self._eps=10e-8
        self._vda=0.0
        self._sda=0.0
        self._vdb=0.0
        self._sdb=0.0

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
        # self._r = np.reshape(self._r, (1,len(self._r)) )
        # self._u = np.reshape(self._u, (1,len(self._u)) )
        # self._y = np.reshape(self._y, (1,len(self._y)) )

        # Compute objective
        self._objective = 0.5 * np.linalg.norm(self._r - self._y)**2

        # compute error
        self._error = self._r - self._y

        # Partial derivatives w.r.t system output, y
        self._dL_dy = -1.0 * (self._r - self._y)

        # Partial derivatives w.r.t controller output, u
        # This is computed numerically using finite differences
        u_shifted = np.roll(self._u,1)
        u_shifted[0]=0.0
        y_shifted = np.roll(self._y,1)
        y_shifted[0]=0.0
        dy_du = y_shifted/u_shifted
        # Handle inf/nan elements (for now, replace nan by 0, inf by a large number and copy sign)
        dy_du = np.nan_to_num(dy_du)

        # Partial derivatives w.r.t conroller numerator coeffs, b
        dtype_np = self._error.dtype
        d0_np = np.array([1.0], dtype=dtype_np)
        n_b = len(self._b) # Number of b coeffs
        T = len(self._error) # Number of time steps
        # Compute forward sensitivities w.r.t. the controller's b_i parameters
        db = np.zeros_like(self._error, shape=(T,n_b)) # [T, n_b]
        db[:, 0] = sp.signal.lfilter(d0_np, self._a, self._error)
        for idx_coeff in range(1, n_b):
            db[idx_coeff:, idx_coeff] = db[:-idx_coeff, 0]

        dL_dy = np.reshape(self._dL_dy, (len(self._dL_dy), 1))
        dy_du = np.reshape(dy_du, (len(dy_du), 1))
        self._dL_db = np.sum(dL_dy * dy_du * db, axis=0)

        # Partial derivatives w.r.t conroller denominator coeffs, a
        # Compute forward sensitivities w.r.t. the controller's a_i parameters
        n_a = len(self._a) # Number of a coeffs
        d1_np = np.array([0.0, 1.0], dtype=dtype_np)
        da = np.zeros_like(self._error, shape=(T,n_a-1)) # [T, n_a-1], remove the 1st element since it's constant, a[0]=1
        da[:, 0] = sp.signal.lfilter(d1_np, self._a, -self._u)
        for idx_coeff in range(1, n_a-1):
            da[idx_coeff:, idx_coeff] = da[:-idx_coeff, 0]

        # Note that the gradient dL_da does not include the one for a[0], because it's constant, a[0]=1
        # dL_da[0] is the derivative of L w.r.t. a[1]
        self._dL_da = np.sum(dL_dy * dy_du * da, axis=0)

        if(self._debug):
            print("\n[DEBUG] [backward] Done with back probagation\n")

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

        if (self._debug):
            print("[DEBUG] [update] ************* Iteration number {} *************".format(iter))

        if ( self.backward() ):
        
            if (self._use_adam):
                # updating a_i
                self._vda = self._beta1*self._vda + (1-self._beta1)*self._dL_da
                self._sda = self._beta2*self._sda + (1-self._beta2)*(self._dL_da**2)

                vda_corrected = self._vda/(1.0-self._beta1**iter)
                sda_corrected = self._sda/(1.0-self._beta2**iter)
                new_a = self._a[1:] - self._alpha * vda_corrected/(np.sqrt(sda_corrected)+self._eps)
                self._new_a = self._a
                self._new_a[1:] = new_a # first element a[0] is always = 1

                
                # updating b_i
                self._vdb = self._beta1*self._vdb + (1-self._beta1)*self._dL_db
                self._sdb = self._beta2*self._sdb + (1-self._beta2)*(self._dL_db**2)

                vdb_corrected = self._vdb/(1.0-self._beta1**iter)
                sdb_corrected = self._sdb/(1.0-self._beta2**iter)
                new_b = self._b - self._alpha * vdb_corrected/(np.sqrt(sdb_corrected)+self._eps)
                self._new_b = new_b

                if(self._debug):
                    print("\n[DEBUG] [update] Done with update step\n")
                    
                return True
            else:
                # TODO Use regular gradient descent algorithm
                print("\n[WARNING] [update] Regular gradient descent is not yet implemented.\n")
                return False

    ################## Setter functions ##################
    def setSignals(self, r=None, u=None, y=None):
        """Setter function for the system signals.
        Updates self._r, self._u, self._y

        @param r Array of system's reference signal
        @param u Array of controller output signal
        @param y Array of system's output signal
        """

        # Sanity checks
        if (r is None):
            print("[ERROR] [setSignals] reference signal r is None.")
            return False
        if (u is None):
            print("[ERROR] [setSignals] Controller output signal u is None.")
            return False
        if (y is None):
            print("[ERROR] [setSignals] System's output signal y is None.")
            return False

        self._r=r
        self._u=u
        self._y=y

        if (self._debug):
            print("[DEBUG] [setSignals] Reference signal , controller & system output signals are set.")
            print("[DEBUG] [setSignals] Signal lengths: len(r)={}, len(u)={}, len(y)={}".format(len(self._r), len(self._u), len(self._y)))

        return True

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

    def setContCoeffs(self, den_coeff=None, num_coeff=None):
        """Setter function for controller coeffs a,b

        @return True if successful, False if not
        """
        if (den_coeff is None):
            print("[ERROR] [setContCoeffs] Denominator (den_coeff) is None.")
            return False
        if (num_coeff is None):
            print("[ERROR] [setContCoeffs] Numerator (num_coeff) is None.")
            return False
        if (den_coeff[0] != 1):
            print("[ERROR] [setContCoeffs] den_coeff[0] should be 1.0 .")
            return False

        self._a = den_coeff
        self._b = num_coeff

    ################ Getter functions ################

    def getNewContCoeff(self):
        """Returns _new_a, _new_b
        """
        return self._new_a, self._new_b

    def getDerivatives(self):
        """Return _dL_dy, _dL_du, _dL_da, _dL_db
        """
        return self._dL_dy, self._dL_du, self._dL_da, self._dL_db

    def getError(self):
        """Returns self._error
        """
        return self._error
    
    def getObjective(self):
        """Returns self._objective
        """
        return self._objective

def test():
    obj = BackProbOptimizer()

if __name__ == "__main__":
    test()