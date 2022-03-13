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
import scipy
import scipy.signal

class BackProbOptimizer:
    """Implements backprobagation techniques to optimize a linear controller for an unkown dynamical system, given desired/reference signal, system & controller output data.
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
        self._new_a=None    # Updated a (denominator) coeffs after performing backward probagation
        self._new_b=None    # Updated b (numerator) coeffs after performing backward probagation

        # Partial derivatives
        self._dL_dy=None    # Partial derivatives w.r.t to system output y
        self._dL_du=None    # Partial derivatives w.r.t to controller output u
        self._dL_da=None    # Partial derivatives w.r.t to controller denomentator's coeffs, a
        self._dL_db=None    # Partial derivatives w.r.t to controller numerator's coeffs, b

        # Maximum numer of iterations
        self._max_iter= None
        # Optimization objective function
        self._objective=None
        # List computed the value of the objective function. Useful for plotting
        self._performance_list = []
        # Max number of objective values, to avoid blowing up the self._performance_list
        self._size_performance_list = 1000
        # Error array between reference signal _r and system output _y
        self._error=None


        # Optimization learning rate
        self._alpha=0.001
        # List of learning rates for different iterations
        self._alpha_list = []
        self._use_optimal_alpha = True
        # Jacobian smalled eigne value
        self._smallest_eig_val=0
        # List of smallest jacobian eigen value in different iterations
        self._eig_val_list = []

        # ADAM optimizer parameters
        self._use_adam=False # True: Use  ADAM algorithm (recommended); False: Use regular gradient descent algorithm
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

        @return True: if successful, False if not
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

        # compute error vector
        self._error = self._r - self._y
        # Compute objective
        self._objective = 0.5 * np.linalg.norm(self._error)**2

        # Partial derivatives w.r.t system output, y
        self._dL_dy = -1.0 * (self._error)
        dL_dy = np.reshape(self._dL_dy, (len(self._dL_dy), 1))

        # Partial derivatives w.r.t controller output, u
        # This is computed numerically using finite differences
        u_shifted = np.roll(self._u,1)
        u_shifted[0]=0.0
        y_shifted = np.roll(self._y,1)
        y_shifted[0]=0.0
        dy_du = (self._y-y_shifted)/(self._u-u_shifted)
        # Handle inf/nan elements (for now, replace nan by 0, inf by a large number and copy sign)
        #dy_du = np.nan_to_num(dy_du, posinf=0.0, neginf=0.0) # only in numpy>= 1.17
        dy_du = np.nan_to_num(dy_du)
        dy_du = np.reshape(dy_du, (len(dy_du), 1))

        # Partial derivatives w.r.t conroller numerator coeffs, b
        dtype_np = self._error.dtype
        d0_np = np.array([1.0], dtype=dtype_np)
        n_b = len(self._b) # Number of b coeffs
        T = len(self._error) # Number of time steps
        # Compute forward sensitivities w.r.t. the controller's b_i parameters
        #db = np.zeros_like(self._error, shape=(T,n_b)) # [T, n_b]
        db = np.zeros( (T,n_b) )
        db[:, 0] = scipy.signal.lfilter(d0_np, self._a, self._error)
        for idx_coeff in range(1, n_b):
            db[idx_coeff:, idx_coeff] = db[:-idx_coeff, 0]
        
        self._dL_db = np.sum(dL_dy * dy_du * db, axis=0)

        # Calculate the Jacobain matrix J_b
        J_b = (dy_du * db).transpose()

        # Partial derivatives w.r.t conroller denominator coeffs, a
        # Compute forward sensitivities w.r.t. the controller's a_i parameters
        n_a = len(self._a) # Number of a coeffs
        d1_np = np.array([0.0, 1.0], dtype=dtype_np)
        #da = np.zeros_like(self._error, shape=(T,n_a-1)) # [T, n_a-1], remove the 1st element since it's constant, a[0]=1
        da = np.zeros((T,n_a-1))
        da[:, 0] = scipy.signal.lfilter(d1_np, self._a, -self._u)
        for idx_coeff in range(1, n_a-1):
            da[idx_coeff:, idx_coeff] = da[:-idx_coeff, 0]

        # Note that the gradient dL_da does not include the one for a[0], because it's constant, a[0]=1
        # dL_da[0] is the derivative of L w.r.t. a[1]
        self._dL_da = np.sum(dL_dy * dy_du * da, axis=0)

        # Calculate the Jacobain matrix J_a
        J_a = (dy_du * da).transpose()

        # Finally construct J
        J = np.vstack((J_a, J_b))

        JJ = -1.0 * np.matmul(J, J.transpose())
        # Absolute value of smallest Eigen value
        (eigVals, eigVec) = np.linalg.eig(JJ)
        lamd = abs(min(eigVals))
        self._smallest_eig_val = lamd
        if(self._debug):
            print("Absolute value of smallest Eigen value of -1*J J^T = {} \n".format(lamd))

        # Optimal learning rate
        alpha = 2./lamd
        if(self._use_optimal_alpha):
            self._alpha = alpha - 0.1*alpha # just subtract a small amount to maintain positive definiteness

        if(self._debug):
            print("Optimal learning rate alpha={}".format(alpha))

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
                self._new_a = np.zeros(len(self._a))
                self._new_a[0]=1.0
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
                # Use regular gradient descent algorithm
                new_a = self._a[1:] - self._alpha * self._dL_da
                self._new_a = np.zeros(len(self._a))
                self._new_a[0]=1.0
                self._new_a[1:] = new_a # first element a[0] is always = 1

                new_b = self._b - self._alpha * self._dL_db
                self._new_b = new_b

                # print("\n[WARNING] [update] Regular gradient descent is not yet implemented.\n")
                return True
        else:
            print("\n[ERROR] [update] Error in backward(). Returned False\n")
            return False

    def maxSensitivity(self, dt=None):
        """Computes the maximum sensitivity of the closed loop discrete system based on
            * Input/output signals (self._u, self._y)
            * Desired frequency range
            * Sampling time
        @param dt sampling time in seconds

        @return s_max Maximum sensitivity
        @return omega_max frequency at maximum sensitivity, rad/s
        @return f_max frequency at maximum sensitivity, Hz
        """
        s_max = None
        omega_max = None
        f_max = None
        ret = (s_max, omega_max, f_max)
        # Sanity checks
        if (dt is None):
            print("\n[ERROR] [maxSensitivity] Sampling time is None\n")
            return ret
        if (self._r is None):
            print("\n[ERROR] [maxSensitivity] Reference signal self._r is None\n")
            return ret
        if (self._u is None):
            print("\n[ERROR] [maxSensitivity] Controller's output signal self._u is None\n")
            return ret
        if (self._y is None):
            print("\n[ERROR] [maxSensitivity] System's output signal self._y is None\n")
            return ret

        # 1- Compute DFT of input/output signals
        n = len(self._u)                                # Signal length. Should be the same as _y, _r
        L = np.arange(1, np.floor(n/2), dtype='int')    # Select first half of the frequencies (symmetry)

        u_fft_full = np.fft.fft(self._u, n)             # Compute FFT
        u_fft = u_fft_full[L]

        y_fft_full = np.fft.fft(self._y, n)
        y_fft = y_fft_full[L]

        freq_full = (1/(dt*n)) * np.arange(n)           # Create array of frequencies
        freq = freq_full[L]
        omega = 2.0 * np.pi * freq                      # Convert from Hz to rad/s, omega = 2 * pi * f
        omega = omega.reshape((1, len(omega)))      # Make it of shape (1,n_f), instead of (n_f,)

        # 2- Estimated plant, \hat{P} at all frequencies
        P_hat = y_fft/u_fft
        P_hat = P_hat.reshape((1,len(P_hat)))

        # 3- compute controller at all frequencies
        a = np.array(self._a)
        b = np.array(self._b)
        cnt_num = b.reshape((len(b),1))     # Controller numerator coefficients. Make it of shape (n_b,1), instead of (n_b,)
        cnt_den = a.reshape((len(a),1))     # Controller denominator coefficients. Make it of shape (n_a,1), instead of (n_a,)

        a_seq = np.arange(0,len(a))               # Used to build exp array [exp(0), exp(1), exp(2), ...]^T
        a_seq = np.reshape(a_seq, (len(a_seq), 1)) * -1j * dt

        b_seq = np.arange(0,len(b))               # [0, 1, 2, ...]^T
        b_seq = np.reshape(b_seq, (len(b_seq), 1)) * -1j * dt   # [0, -j*dt, -2j*dt, ...]^T

        a_mat = a_seq * omega                           # Controller denominator. Each column vector corersonds to the controller denominator terms at specific omega
                                                        # Example, each column vector a_mat[:,0] = [0, -j*dt*omega_0, -2j*dt*omega_0, ...]^T
        b_mat = b_seq * omega

        a_mat_exp = np.exp(a_mat)                       # Each column: [1.0, exp(-j*w_i*dt), exp(-2j*w_i*dt), ...]^T
        a_mat_exp = cnt_den * a_mat_exp                 # Each column: [1.0, a_1*exp(-j*w_i*dt), a_2*exp(-2j*w_i*dt), ...]^T
        b_mat_exp = np.exp(b_mat)                       # Each column: [1.0, exp(-j*w_i*dt), exp(-2j*w_i*dt), ...]^T
        b_mat_exp = cnt_num * b_mat_exp                 # Each column: [b_0, b_1*exp(-j*w_i*dt), b_2*exp(-2j*w_i*dt), ...]^T

        a_omega = np.sum(a_mat_exp, axis=0)             # Summ all the rows. The result is the controller's denominator at all omegas
        b_omega = np.sum(b_mat_exp, axis=0)             # Summ all the rows. The result is the controller's numerator at all omegas
        cnt_omega = b_omega / a_omega                   # Finally, compute controller at all omegas, a row vector

        # 4- Compute the maximum sensitivity
        s = 1.0 / (1.0 + cnt_omega*P_hat)
        s_abs = np.abs(s)                               # Magnitude of the sensitivity function, at all omegas       
        s_argmax = np.argmax(s_abs)                     # The index of the maximum sensitivity
        s_max = s_abs[0][s_argmax]                      # Maximum sensitivity
        omega_max = omega[0][s_argmax]                  # Frequency at max sensitivity, rad/s
        f_max = freq[s_argmax]                          # Frequency at max sensitivity, Hz

        ret=(s_max, omega_max, f_max)

        return ret


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

    def updateAlphaList(self):
        self._alpha_list.append(self._alpha)

    def resetAlphaList(self):
        self._alpha_list = []

    def updatePerformanceList(self):
        self._performance_list.append(self._objective)

    def resetPerformanceList(self):
        self._performance_list = []

    def updateEigValList(self):
        self._eig_val_list.append(self._smallest_eig_val)
    
    def resetEigValList(self):
        self._eig_val_list= []

    def resetLists(self):
        """Resets all lists
        """
        self.resetAlphaList()
        self.resetEigValList()
        self.resetPerformanceList()


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
