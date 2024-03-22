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
    LeadLage_RT(MV, Kp, Tlead, Tlag, Ts, PV, PVInit=0, method='EBD')
        The function "LeadLag_RT" needs to be included in a "for or while loop".

        :MV: input vector
        :Kp: process gain
        :Tlead: lead time constant [s]
        :Tlag: lag time constant [s]
        :Ts: sampling period [s]
        :PV: output vector
        :PVInit: (optional : default value is 0)
        :method: discretisation method (optional : default value is 'EBD')
            EBD: Euler Backward Difference
            EFD: Euler Forward Difference
            TRAP: Trapezoïdal method

        The function appends a value to the output vectot "PV".
        The appended value is obtained from a recurrent equation that depends on the discretisation method.
    
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
def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, method='EBD-EBD'):
    
    """
    The function "PID_RT" needs to be included in a "for or while loop".
    
    :SP: SP (or SetPoint) vector
    :PV: PV (or process Value) vector
    :Man: Man (or Manual value for MV) vector (True or False)
    :MVMan: MVMan (or manual value for MV) vector
    :MVFF: MVFF (or Feedforward) vector
    
    :Kc: controller gain
    :Ti: integral time constant [s]
    :Td: derivate time constant [s]
    :alpha: Tfd = alpha*Td where Tfd is the derivative filter time constant [s]
    
    :MVMin: minimum value for MV (used for saturation and anti wind-up)
    :MVMax: maximum value for MV (usde for saturation and anti wind-up)
    
    :MV: MV (or Manipulated Value) vector
    :MVP: MVP (or Proportional part of MV) vector
    :MVI: MVI (or Intefral part of MV) vector
    :MVD: MVD (or Derivative part of MV) vector
    :E: E (or control Error) vector
    
    :ManFF: Activated FF in manual mode (optional: default boolean value is False)
    :PVInit: Initial value for PV (optional: default value is 0): used if PID_RT is ran first in the sequence and no value of PV is available yet.
    
    :method: discretisation method (optional: default value is 'EBD')
    EBD-EBD: EBD for integral action and EBD for derivative action
    EBD-TRAP: EBD for integral action and TRAP for derivative action
    TRAP-EBD: TRAP for integral action and EBD for derivative action
    TRAP-TRAP: TRAP for integral action and TRAP for derivative action
    
The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI" and "MVD".
The appended values are based on the PID algorithm, the controller mode, and feedforward. Note that the saturation of "MV" within the limits [MVMin MVMax] is implemented with anti wind-up.
"""
    # MV[k] is MV[-1] and MV[k-1] is MV[-2] because we're creating it in real time
    
    Tfd = alpha*Td

    methods = method.split("-")
    methodI = methods[0]
    methodD = methods[1]
    
    #Init of E
    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else:
        E.append(SP[-1] - PV[-1])
    
    #Init of MVI
    if len(MVI) == 0:
        MVI.append((Kc*Ts/Ti)*E[-1])
    else:
        if methodI == 'TRAP':
            MVI.append(MVI[-1] + (0.5*Kc*Ts/Ti)*(E[-1]+E[-2]))
        else: #'EBD'
            MVI.append(MVI[-1] + ((Kc*Ts/Ti)*E[-1] ))
            
    #Init of MVD
    if len(MVD) == 0:
        MVD.append(((Kc*Td)/(Tfd+Ts/2))*(E[-1]))
    else:
        if methodD == 'TRAP':
            MVD.append(MVI[-1] + (0.5*Kc*Ts/Ti)*(E[-1]+E[-2]))
        else: #'EBD'
            MVD.append((Tfd-Ts/2)/(Tfd+Ts/2)*MVD[-1]+((Kc*Td)/(Tfd+Ts/2))*(E[-1]-E[-2]))
                       
    #Mode Automatic        
    if (Man[-1] == False & len(MVI)>=2): 
        MVP.append(Kc*E[-1])
        MVI.append(MVI[-2] + ((Kc*Ts)/Ti) * E[-1]) #MVI[-2] ?
        MVD.append((Tfd-Ts/2)/(Tfd+Ts/2)*MVD[-2]+((Kc*Td)/(Tfd+Ts/2))*(E[-1]-E[-2]))
        MV.append(MVP[-1]+MVI[-1]+MVD[-1])
                       
    #Manual Mode + Anti Wind-up
    elif(Man[-1] == True):
        if ManFF:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]
        else:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]
                       
        MV.append(MVMan)
    
    #Anti Saturation Mechanism
    if(len(MVP)>=1):
        if ((MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1]) > MVMax):
            MVI[-1] = MVP[-1] - MVD[-1] - MVFF[-1] - MVMax
        elif ((MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1]) < MVMin):
            MVI[-1] = MVP[-1] - MVD[-1] - MVFF[-1] - MVMax

    return(MV[-1])
        