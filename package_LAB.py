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

    LeadLag_RT(MV, Kp, Tlead, Tlag, Ts, PV, PVInit=0, method='EBD')
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
    if (Tlead != 0 and Tlag != 0):
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
    MVToAppend = SP[-1]

    methods = method.split("-")
    methodI = methods[0]
    methodD = methods[1]
    
    #Init of E
    if len(PV) == 0:
        E.append(SP[-1] - PVInit)    
    else:
        E.append(SP[-1] - PV[-1])

    
    #Init of MVP
    if len(MVP) == 0:
        MVP.append(0)
    
    #Init of MVI
    if len(MVI) == 0:
        MVI.append((Kc*Ts/Ti)*E[-1])
    
    #elif len(MVI) == 1 :
    else :
        if methodI == 'TRAP':
            MVI.append(MVI[-1] + (0.5*Kc*Ts/Ti)*(E[-1]+E[-2]))
        else: #'EBD'
            MVI.append(MVI[-1] + ((Kc*Ts/Ti)*E[-1] ))
        
    #Init of MVD
    if len(MVD) == 0:
        MVD.append(((Kc*Td)/(Tfd+Ts/2))*(E[-1]))
    #elif len(MVD) == 1:
    else:
        if methodD == 'TRAP':
            MVD.append(MVI[-1] + (0.5*Kc*Ts/Ti)*(E[-1]+E[-2]))
        else: #'EBD'
            MVD.append((Tfd-Ts/2)/(Tfd+Ts/2)*MVD[-1]+((Kc*Td)/(Tfd+Ts/2))*(E[-1]-E[-2]))

    
    #Mode Automatic        
    if (Man[-1] == False and len(MVI)>=2): 
        
        MVP.append(Kc*E[-1])
        MVITemp = (MVI[-2] + ((Kc*Ts)/Ti) * E[-1])
        MVDTemp = ((Tfd-Ts/2)/(Tfd+Ts/2)*MVD[-2]+((Kc*Td)/(Tfd+Ts/2))*(E[-1]-E[-2]))
        MVToAppend = MVP[-1] + MVITemp + MVDTemp + MVFF[-1]
   
                       
    #Manual Mode + Anti Wind-up
    elif(Man[-1] == True):
        if ManFF:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]

        else:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]
        
        MVToAppend = MVMan[-1]
        MVP.append(0)
    
            
    #Anti Saturation Mechanism
    if (MVToAppend > MVMax):      #Max
        MVI[-1] = - MVP[-1] - MVD[-1] - MVFF[-1] + MVMax 
      
        if (Man[-1] == False):
            MVToAppend = MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1]
            
    elif (MVToAppend < MVMin):    #Min
        MVI[-1] =  MVMin - MVP[-1] - MVD[-1] - MVFF[-1]
        if (Man[-1] == False):
            MVToAppend = MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1]

        
    if(len(SP) >= 2):
        MV.append(MVToAppend)
        
        
#-----------------------------------       
def Margin(P, omega, Am=0, Phim=0, omegaC=0, omegaU=0, Show=True):
    
    """
    :P: Process as defined by the class "Process".
        Use the following command to define the default process which is simply a unit gain process:
            P = Process({})
        
        A delay, two lead time constants and 2 lag constants can be added.
        
        Use the following commands for a SOPDT process:
            P.parameters['Kp'] = 1.1
            P.parameters['Tlag1'] = 10.0
            P.parameters['Tlag2'] = 2.0
            P.parameters['theta'] = 2.0
        
        Use the following commands for a unit gain Lead-lag process:
            P.parameters['Tlag1'] = 10.0        
            P.parameters['Tlead1'] = 15.0        
        
    :omega: frequency vector (rad/s); generated by a command of the type "omega = np.logspace(-2, 2, 10000)".
    :Show: boolean value (optional: default value = True).
        If Show == True, the Bode diagram is show.
        If Show == False, the Bode diagram is NOT show and the complex vector Ps is returned.
    
    The function "Bode" generates the Bode diagram of the process P
    """     
    
    s = 1j*omega
    
    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    PLag1 = 1/(P.parameters['Tlag1']*s + 1)
    PLag2 = 1/(P.parameters['Tlag2']*s + 1)
    PLead1 = P.parameters['Tlead1']*s + 1
    PLead2 = P.parameters['Tlead2']*s + 1
    
    Ps = np.multiply(Ptheta,PGain)
    Ps = np.multiply(Ps,PLag1)
    Ps = np.multiply(Ps,PLag2)
    Ps = np.multiply(Ps,PLead1)
    Ps = np.multiply(Ps,PLead2)
    
    
    omegaC = None
    index_omegaC = None
    for i in range(len(omega) - 1):
        if np.sign(20*np.log10(np.abs(Ps[i]))) != np.sign(20*np.log10(np.abs(Ps[i+1]))):
            omegaC = omega[i]
            index_omegaC = i
            break
            
            
    # Calculate phase values
    phase_degrees = (180/np.pi)*np.unwrap(np.angle(Ps))

    # Find the index where phase is closest to -180 degrees
    index_omegaU = np.argmin(np.abs(phase_degrees + 180))

    # Get the corresponding frequency
    omegaU = omega[index_omegaU]            

        
    if Show == True:
        
        fig, (ax_gain, ax_phase) = plt.subplots(2,1)
        fig.set_figheight(12)
        fig.set_figwidth(22) 
        
        
        yminOmegaU= 20*np.log10(np.abs(Ps[index_omegaU]))
        ymaxOmegaU= 0
        
        yminOmegaC = -180
        ymaxOmegaC = (180/np.pi)*np.angle(Ps[index_omegaC])
        
        displayAmY = (ymaxOmegaU+yminOmegaU)/2
        displayAmX = omegaU+(0.01)
        
        displayPhiMY = (ymaxOmegaC+yminOmegaC)/2
        displayPhiMX = omegaC - 0.0025
        
        Am = np.abs(ymaxOmegaU - yminOmegaU)
        
        Phim = np.abs(ymaxOmegaC-yminOmegaC)


        # Gain part
        ax_gain.semilogx(omega,20*np.log10(np.abs(Ps)),label=r'$P(s)$')   
        gain_min = np.min(20*np.log10(np.abs(Ps/5)))
        gain_max = np.max(20*np.log10(np.abs(Ps*5)))
        ax_gain.set_xlim([np.min(omega), np.max(omega)])
        ax_gain.set_ylim([gain_min, gain_max])
        ax_gain.axhline(y=0, color='green',linestyle='-')
        ax_gain.axvline(x=omegaC, color='red', linestyle=':')
        ax_gain.axvline(x=omegaU, color='lime', linestyle=':')
        ax_gain.vlines(x=omegaU, ymin=yminOmegaU, ymax=ymaxOmegaU, color='#11aa00', linestyle='solid', linewidth=5)
        ax_gain.text(displayAmX, displayAmY,'20 log$_{10}$ A$_{m}$', color='black', fontsize=15, ha='left')
        ax_gain.set_ylabel('Amplitude' + '\n $|P(j\omega)|$ [dB]')
        ax_gain.set_title('Bode plot of P')
        ax_gain.text(omegaC, gain_min + 2,'$\omega_{c}$', color='black', fontsize=15, ha='left')
        ax_gain.text(omegaU, gain_min + 2,'$\omega_{u}$', color='black', fontsize=15, ha='left')
        ax_gain.legend(loc='best')
    
        # Phase part
        ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ps)),label=r'$P(s)$')   
        ax_phase.set_xlim([np.min(omega), np.max(omega)])
        ph_min = np.min((180/np.pi)*np.unwrap(np.angle(Ps))) - 10
        ph_max = np.max((180/np.pi)*np.unwrap(np.angle(Ps))) + 10
        ax_phase.axhline(y=-180, color='green',linestyle='-')
        ax_phase.axvline(x=omegaU, color='lime', linestyle=':')
        ax_phase.axvline(x=omegaC, color='red', linestyle=':')
        ax_phase.set_ylim([np.max([ph_min, -200]), ph_max])
        ax_phase.vlines(x=omegaC, ymin=yminOmegaC, ymax=ymaxOmegaC, color=[0.8, 0, 0 , 0.8], linestyle='solid', linewidth=5)
        ax_phase.text(displayPhiMX, displayPhiMY,'$\phi_{m}$', color='black', fontsize=15, ha='left')
        ax_phase.set_xlabel(r'Frequency $\omega$ [rad/s]')        
        ax_phase.set_ylabel('Phase' + '\n $\,$'  + r'$\angle P(j\omega)$ [°]')
        ax_phase.legend(loc='best')        
        
        return Am, Phim
        

    else:
        return Ps
    
    
#-----------------------------------       

def IMC_Tuning(T1, T2, T1p, gamma, Kp):
        
    """
    :P: Process as defined by the class "Process".
        Use the following command to define the default process which is simply a unit gain process:
            P = Process({})
        
        A delay, two lead time constants and 2 lag constants can be added.
        
        Use the following commands for a SOPDT process:
            P.parameters['Kp'] = 1.1
            P.parameters['Tlag1'] = 10.0
            P.parameters['Tlag2'] = 2.0
            P.parameters['theta'] = 2.0
        
        Use the following commands for a unit gain Lead-lag process:
            P.parameters['Tlag1'] = 10.0        
            P.parameters['Tlead1'] = 15.0        
        
    :omega: frequency vector (rad/s); generated by a command of the type "omega = np.logspace(-2, 2, 10000)".
    :Show: boolean value (optional: default value = True).
        If Show == True, the Bode diagram is show.
        If Show == False, the Bode diagram is NOT show and the complex vector Ps is returned.
    
    The function "Bode" generates the Bode diagram of the process P
    """     
    
    Tclp = gamma * T1p
    
    Kc = ((T1 + T2) / Tclp) / Kp
    
    Ti = T1 + T2
    
    Td = (T1 * T2) / T1 + T2
    
    return [Kc, Ti, Td, Tclp] 
