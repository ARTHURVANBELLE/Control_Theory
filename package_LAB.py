import pandas as pd
import numpy as np
import math
import package_DBR

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from package_DBR import myRound, SelectPath_RT, Delay_RT, FO_RT, FOPDT, SOPDT, FOPDT_cost, SOPDT_cost, Process, Bode

#-----------------------------------
def LeadLag_RT(MV, Kp, Tlead, Tlag, Ts, PV, PVInit=0, method='EBD'):
    
    """
    L
    
    """
    if (Tlead & Tlag != 0):
        K = Ts/Tlag
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2] and PV[k] = PV[-1]
            if method == 'EBD':
                PV.append(1/(1+K)*PV[-1] + ((Kp*K)/(1+K))*(((1+(Tlead/Ts))*MV[-1])-((Tlead/Ts)*MV[-2])))
            elif method == 'EFD':
                PV.append((1-K)*PV[-1] + Kp*K*((Tlead/Ts)*MV[-1]+(1-(Tlead/Ts))*MV[-2])) 
            #elif method == 'TRAP':
            #    PV.append((1/(2*T+Ts))*((2*T-Ts)*PV[-1] + Kp*Ts*(MV[-1] + MV[-2])))            
            else:
                PV.append(1/(1+K)*PV[-1] + ((Kp*K)/(1+K))*(((1+(Tlead/Ts))*MV[-1])-((Tlead/Ts)*MV[-2])))
                
    elif (Tlead != 0):
        return(FO_RT(MV, Kp, Tlag, Ts, PV))
        
    elif (Tlag != 0):
        return(FO_RT(MV, Kp, Tlead, Ts, PV))
               
    else:
        PV.append(Kp*MV[-1])

    
    return ()

#-----------------------------------
def PID_Controller():
    
    """
    
    """
    
    return ()
