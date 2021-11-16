from matplotlib.legend import Legend
import matplotlib.pyplot as plt
from octune.optimization import BackProbOptimizer
from octune.control_system import ControlSystem

def main():
    ############ Common variables ############
    dt = 0.01       # Time sample [seconds]
    max_time = 2.0  # Maximum time [seconds]
    step = 1.0      # step input value
    max_iter = 1000 # Maximum number of optimization iterations


    ############ Controller ############
    cnt=ControlSystem()
    cnt.setDebug(val=False)
    
    # Build a stable discrete plant with two real poles within the unit circle
    # denominator z^2+(pole1+pole2)z+(pole1*pole2
    pole1=0.4
    pole2=-0.3
    a0=1.0
    a1=pole1+pole2
    a2=pole1*pole2
    cnt.buildPlantTF(num_coeff=[1], den_coeff=[a0,a1,a2])

    # Discrete PID controller
    kp=0.3
    ki=1
    kd=0.0001
    # Numerator coeff
    n0=kp+ki*dt/2.+kd/dt
    n1=-kp+ki*dt/2.-2.*kd/dt
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
    init_r=r
    init_u=u
    init_y=y
    print("Type of y {}, type of u {}".format(type(y),type(u)))
    

    ############ For plotting ############
    performance=[] # Optimization objective function
    kp_list=[kp]
    ki_list=[ki]
    kd_list=[kd]

    ############ Optimization ############
    opt=BackProbOptimizer()
    opt._debug=True
    # Initial data
    opt.setSignals(r=r,u=u,y=y)
    opt.setContCoeffs(den_coeff=[1,-1,0], num_coeff=[n0,n1,n2])

    for i in range(1,max_iter+1):
        if(opt.update(iter=i)):
            performance.append(opt.getObjective())

            # get new conrtroller coeffs
            den,num=opt.getNewContCoeff()
            kp,ki,kd=cnt.getPIDGainsFromCoeff(num)
            kp_list.append(kp)
            ki_list.append(ki)
            kd_list.append(kd)

            # update controller (practically, this would be updated on hardware)
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
        else:
            print("\n[ERROR] [main] Could not perform update step\n")
            break

    final_u=u
    final_y=y
    ################### Plotting ###################
    plt.plot(t,init_r, 'r', label='Desired reference')
    plt.plot(t,init_y, 'k', label='Initial response')
    plt.plot(t,final_y, 'g', label='Tuned response')
    plt.xlabel('time [seconds]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.show()

    plt.plot(performance)
    plt.xlabel('iteration number')
    plt.ylabel('performance')
    plt.show()

    plt.plot(kp_list, label='Kp')
    plt.plot(ki_list, label='Ki')
    plt.plot(kd_list, label='Kd')
    plt.xlabel('iteration number')
    plt.ylabel('Gain value')
    plt.legend()
    plt.show()


if __name__=="__main__":
    main()