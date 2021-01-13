import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


"""
Modified from:
http://apmonitor.com/pdc/index.php/Main/ProportionalIntegralControl
"""

class FOPDT:
    """
    First Order Process Plus Dead Time model
    """

    def __init__(self, Kp=None, τp=None, θp=None):
        
        # Process Gain (Δy/Δu)
        self.Kp = Kp

        # Process Time constant (process goes from one ss to 63.2%  of other ss in one τ)   
        self.τp = τp
        
        # Process Dead time (time it takes the output to respond to the input)
        self.θp = θp

    def process(self, t, y, u):
        y = y[0] # Extract values
        dydt = (-y / self.τp) + (self.Kp/self.τp) * u
        return [dydt]


class ControllerPID:
    def __init__(self, Kc=None, τI=None):
        # Controller's Proportional gain
        self.Kc = Kc

        # Controller's integral time constant
        self.τI = τI

        self.output_max = 100
        self.output_min = 0

        self.name = ''
        if Kc is not None:
            self.name += 'P'
        if (Kc is not None) and (τI is not None):
            self.name += 'I'


class Simulation:
    def __init__(self, setpoint, t_sim=1200, t_res=10, mode=0):
        """
        Args:
            setpoint : Vector defining the setpoint
            t_sim    : Total simulation time
            t_res    : Simulation resolution (ticks per second)
            mode     : 0: open loop; 1: closed loop
        """

        # Total simulation time
        self.t_sim = t_sim

        # Time resolution (ticks per second)
        self.t_res = t_res

        # Vector with simulation ticks
        self.t = np.linspace(0, t_sim, t_sim * t_res + 1)
        
        # Length of one time step
        self.dt = self.t[1] - self.t[0]

        # Open or closed loop simulation
        self.mode = mode

        # Vector defining the setpoint
        self.sp = setpoint

        # Controller Output storage
        # Open loop (response to a step response)
        if mode == 0:    
            self.co = setpoint
        # Closed loop
        else:
            self.co = np.zeros(t_sim * t_res + 1)

        # Process Variable storage
        self.pv = np.zeros(t_sim * t_res + 1)

        # Error storage (setpoint - process variable)
        self.error = np.zeros(t_sim * t_res + 1)

        # Integral of error
        self.ierror = np.zeros(t_sim * t_res + 1)

    
    def run(self, model, controller):
        if self.mode == 1:
            self.description = f"Closed loop control - {controller.name}"
            if controller.Kc is not None:
                self.description += f' Kc={controller.Kc:.1f} '
            if (controller.Kc is not None) and (controller.τI is not None):
                self.description += f' τI={controller.τI:.1f} '
        else:
            self.description = "Open Loop control"


        for i in range(len(self.t) - 1):
            
            # Store the error in the ith slot 
            self.error[i] = self.sp[i] - self.pv[i]
            
            # Store accumulated error in the ith slot 
            if i > 0:
                self.ierror[i] = self.ierror[i-1] + self.error[i] * self.dt
            
            # Calculate PID controller output terms
            if controller.Kc is not None:
                P = controller.Kc * self.error[i]
            else:
                P = 0
            
            if (controller.Kc is not None) and (controller.τI is not None):
                I = (controller.Kc / controller.τI) * self.ierror[i]
            else:
                I = 0
            
            if self.mode == 1: # Closed loop
                self.co[i] = self.co[0] + P + I # The first term is the bias

            if self.co[i] > controller.output_max:
                self.co[i] = controller.output_max
                # Anti integration windup
                self.ierror[i] -= self.error[i] * self.dt
            
            if self.co[i] < controller.output_min:
                self.co[i] = controller.output_min
                # Anti integration windup
                self.ierror[i] -= self.error[i] * self.dt
            
            # Simulate time delay
            ndelay = int(np.ceil(model.θp / self.dt))
            # Index of the lagged controller output
            iu = max(0, i - ndelay)

            # Prepare parameters for the solver
            t0, tf = self.t[i], self.t[i+1]
            # Lagged controller output, model parameters
            inputs = [self.co[iu]]

            sol = solve_ivp(
                fun      = model.process, # A callable defined above
                t_span   = (t0, tf),      # Interval of integration
                y0       = [self.pv[i], ],# Initial conditions
                method   = 'LSODA',       # 'LSODA': equivalent to odeint;
                args     = inputs         # Additional arguments for the model
            )
            
            # Store the output in the i+1 slot
            self.pv[i+1] = sol.y[0, -1]



    def plot_results(self):

        plt.figure(1, figsize=(12,5))

        # Output
        plt.subplot(4,1,1)
        plt.plot(self.t, self.pv,  'r-', linewidth=3, label='Process Variable (PV)')
        if self.mode == 1:
            plt.plot(self.t, self.sp, 'k:', linewidth=3, label='Setpoint (SP)')
        if self.mode == 0:
            plt.title('Open loop control, step response')
        else:
            plt.title(self.description)

        plt.ylabel('Process output')
        plt.legend()

        # Input
        plt.subplot(4,1,2)
        plt.plot(self.t, self.co, 'b-', linewidth=1, label='Controller Output (OP)')
        plt.ylabel('Process input')    
        plt.legend()

        # Error
        plt.subplot(4,1,3)
        plt.plot(self.t, self.error, 'g-', linewidth=3, label='Error')
        plt.ylabel('Error')
        plt.xlabel('Time (sec)')
        plt.legend()

        # Integral of error
        plt.subplot(4,1,4)
        plt.plot(self.t, self.ierror, 'g-', linewidth=3, label='Integral of error')
        plt.ylabel('Integral of error')
        plt.xlabel('Time (sec)')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    t_sim = 1200
    t_res = 10
    mode = 1
    
    # Define setpoint values
    setpoint = np.zeros(t_sim * t_res + 1)
    setpoint[50*t_res: 600*t_res] = 10

    simulation = Simulation(setpoint, t_sim, t_res, mode)
    model      = FOPDT(Kp=2, τp=200, θp=100)
    controller = ControllerPID(Kc=2, τI=200)

    simulation.run(model, controller)
    simulation.plot_results()