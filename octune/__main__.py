from octune.optimization import BackProbOptimizer
from octune.control_system import ControlSystem

def main():
    ############ Common variables ############
    dt = 0.01       # Time sample [seconds]
    max_time = 2.0  # Maximum time [seconds]
    step = 1.0      # step reference
    max_iter = 1000 # Maximum number of optimization iterations


    ############ Controller ############
    cnt=ControlSystem()
    cnt.setDebug(val=dt)
    
    # Build stable discrete plant
    # denominator z^2+(pole1+pole2)z+(pole1*pole2)
    pole1=0.6
    pole2=0.2
    a0=1.0
    a1=pole1+pole2
    a2=pole1*pole2
    cnt.buildPlantTF(num_coeff=[1], den_coeff=[a0,a1,a2])

    # Discrete PID controller
    kp=0.09
    ki=20
    kd=0.0001
    # Numerator coeff
    n0=kp+ki*dt/2+kd/dt
    n1=-kp+ki*dt/2-2*kd/dt
    n2=kd/dt
    # Denominator coeff
    d0=1.0
    d1=-1.0
    d2=0.0
    cnt.buildControllerTF(num_coeff=[n0,n1,n2], den_coeff=[d0,d1,d2])

    cnt.buildSystemTF()
    cnt.createStepInput(step=1.0, T=max_time)
    cnt.simulateSys()
    t,r,u,y=cnt.getSignals()
    print("Type of y {}, type of u {}".format(type(y),type(u)))
    

    ############ For plotting ############


    ############ Optimization ############
    opt=BackProbOptimizer()
    # Initial data
    opt.setSignals(r=r,u=u,y=y)
    opt.setContCoeffs(den_coeff=[1,-1,0], num_coeff=[n0,n1,n2])

    for i in range(1,max_iter+1):
        opt.update(iter=i)

        # get new conrtroller coeffs
        den,num=opt.getNewContCoeff()

        # update controller
        cnt.buildControllerTF(num_coeff=num, den_coeff=[1,-1,0])
        # build system TF
        cnt.buildSystemTF()
        # simulate system
        cnt.simulateSys()
        # get new signals
        t,r,u,y=cnt.getSignals()

        # Update optimizer
        opt.setSignals(r=r,u=u,y=y)
        opt.setContCoeffs(den_coeff=[1,-1,0], num_coeff=num)

if __name__=="__main__":
    main()