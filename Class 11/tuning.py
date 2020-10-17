import numpy as np


def get_tuning_correlation(Kp, θp, τp, method):
    """

    Tuning correlation for P-only control using the ITAE 
    (Integral of Time-weighted Absolute Error) method. 
    
    Args:
        method: Method used to calculate the tuning correlation: 
            - 'disturbance_rejection' (also referred to as regulatory control)
            - 'setpoint_tracking' (also known as servo control)
        Kp: Process Gain
        θp: Process Dead time
        τp: Process time constant
    """

    if method == 'disturbance_rejection':
        return (0.5 / Kp) * np.power(τp / θp, 1.08)

    elif method == 'setpoint_tracking':
        return (0.2 / Kp) * np.power(τp / θp, 1.22)


