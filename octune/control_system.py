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
import control.matlab as mt
import matplotlib.pyplot as plt

class ControlSystem:
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
        self._time=None         # Array of time sequence

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

        self._plant_tf = cnt.tf(num_coeff, den_coeff, self._dt)

        if (self._debug):
            print("[DEBUG] [buildPlantTF] Plant transfer funciton is created {}".format(self._plant_tf))

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

        self._cont_tf = cnt.tf(num_coeff, den_coeff, self._dt)

        if (self._debug):
            print("[DEBUG] [buildControllerTF] Controller transfer funciton is created {}".format(self._cont_tf))

        return True

    def buildSystemTF(self):
        """Builds closed loop system's transfer function given _plant/controller TFs

        Uses self._cont_tf, self._plant_tf

        Updates self._sys_tf
        """
        # Snaity checks
        if (self._plant_tf is None):
            print("[ERROR] [buildSystemTF] Plant transfer function is None.")
            return False
        if (self._cont_tf is None):
            print("[ERROR] [buildSystemTF] Controller transfer function is None.")
            return False

        self._sys_tf = cnt.feedback(self._cont_tf, self._plant_tf)

        if (self._debug):
            print("[DEBUG] [buildSystemTF] Closed loop system transfer funciton is created.")

        return True

    def simulateSys(self):
        """Simulates the closed loop system by applying self._r
        """
        if(self.buildSystemTF()):
            N = len(self._r)
            t = self._dt*np.arange(N)
            self._y, self._time,x = mt.lsim(self._sys_tf, self._r, t)

            # compute controller output, u
            e = self._r - self._y
            self._u, _, _ = mt.lsim(self._cont_tf, e, t)


            if(self._debug):
                print("\n[DEBUG] building system TF succeedded\n")

            return True
        else:
            print("\n[ERROR] building system TF failed\n")
            return False

    ###################### Setter functions ######################

    def setR(self, r=None):
        """Set system's reference signal, _r

        @r array of reference signal
        """
        if(r is None):
            print("[ERROR] [setR] reference signal is None")

        self._r = r
        if (self._debug):
            print("[DEBUG] [setR] System reference signal is set")

    def setU(self, u=None):
        """Set controller output signal, _u

        @u array of controller output signal
        """
        if(u is None):
            print("[ERROR] [setU] controller output signal is None")
            
        self._u = u
        if (self._debug):
            print("[DEBUG] [setU] Controller output signal is set")

    def setY(self, y=None):
        """Set System output signal, _y

        @y array of system output signal
        """
        if(y is None):
            print("[ERROR] [setY] System output signal is None")

        self._y = y
        if (self._debug):
            print("[DEBUG] [setY] System output signal is set")

    ############################ Getter functions ############################

    def getR(self):
        return self._r

    def getU(self):
        return self._u

    def getY(self):
        return self._y
    
    def getTime(self):
        return self._time

    def getSignals(self):
        return self._time, self._r, self._u, self._y

    ############################## Helper functions ##############################

    def createStepInput(self, step=1.0, T=2.0):
        """Creates a step signal for a duration of T seconds
        Uses self._dt
        Updates self._r

        @param step Signal amplitude
        @param T Time length in seconds
        """
        N = int(np.ceil(T/self._dt))    # Number of samples
        self._r = step * np.ones(N)
        self._r[0] = 0.0

        if (self._debug):
            print("[DEBUG] [createStepInput] Step reference signal of step value {} is created".format(step))

        return True

    def createPeriodicInput(self, mag=1.0, T=2.0):
        """Creates a sin wave for the reference signal
        Uses self._dt
        Updates self._r

        @param mag Signal amplitude
        @param T Time length in seconds
        """
        N = int(np.ceil(T/self._dt))    # Number of samples
        t = self._dt*np.arange(N)
        self._r = np.sin(t)

        if (self._debug):
            print("[DEBUG] [createPeriodicInput] Periodic (sin) reference signal of amplitude {} is created".format(mag))
        
        return True

    def setDebug(self, val=True):
        """Sets _debug to True or False for printing debug messages
        """
        self._debug=val
        print("[DEBUG] Debug mode is set to {}".format(val))

    def setDt(self, dt=0.01):
        """Sets time sample _dt
        """
        if (dt<0):
            print("[ERROR] [setDt] dt should be >0.")
            return False

        self._dt = dt
        if(self._debug):
            print("[DEBUG] [setDt] time sample dt={}".format(dt))

        return True

    def getPIDGainsFromCoeff(self,n0, n1, n2):
        """Computes Kp,Ki,Kd of a discrete PID controller from its transfer function's numerator coeff
        Numerator is of the form n0*Z^2+n1*Z+n2

        @return kp,ki,kd
        """
        if(not self._dt>0):
            print("[ERROR] [getPIDGainsFromCoeff] sampling time should be >0")
            return

        dt=self._dt

        kd=dt*n2
        ki=(n0+n1+n2)/dt
        kp=(n0+n1-n2)/2

        return kp,ki,kd

    def getPIDCoeffFromGains(self, kp, ki, kd):
        """Computes transfer function's numerator of a discrete PID from its gains.
        The denominator is constant Z^2+Z+0
        The numnerator is of the form n0*Z^2+n1*Z+n2

        @return n0,n1,n2
        """
        dt=self._dt
        # Numerator coeff
        n0=kp+ki*dt/2+kd/dt
        n1=-kp+ki*dt/2-2*kd/dt
        n2=kd/dt
        return n0,n1,n2

def test():
    obj=ControlSystem()
    dt=0.01
    obj.setDt(dt)

    obj.setDebug(True)
    #obj.createPeriodicInput(mag=2.0, T=2.0)
    obj.createStepInput(step=1.0, T=2.0)
    obj.buildPlantTF(num_coeff=[1],den_coeff=[1,1.4,0.45])

    # Discrete PID TF
    kp=0.09
    ki=20
    kd=0.0001
    # Numerator coeff
    b0=kp+ki*dt/2+kd/dt
    b1=-kp+ki*dt/2-2*kd/dt
    b2=kd/dt
    # Denominator coeff
    a0=1.0
    a1=-1.0
    a2=0.0
    obj.buildControllerTF(num_coeff=[b0,b1,b2], den_coeff=[a0,a1,a2])

    obj.buildSystemTF()
    obj.simulateSys()
    t,r,u,y=obj.getSignals()

    plt.plot(t,r, 'r')
    plt.plot(t,u, 'k')
    plt.plot(t,y, 'g')
    plt.show()


if __name__== "__main__":
    test()