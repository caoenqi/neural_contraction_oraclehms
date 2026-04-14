import jax
import jax.numpy as jnp
from numpy.linalg import cholesky
import numpy as onp
import immrax as irx
from control import lqr
from pathlib import Path
import csv

g = 9.81

BASE = Path(".")

NCM = BASE / "NCM"
CONTROLLER = BASE / "Controller"


class HRPropofol(irx.System):
    def __init__ (self, params, scaling=1.0) :
        self.xlen = 11
        self.evolution = 'continuous'
        self.scaling = scaling # should either be a single float or an array of length 11 for state-wise scaling

        # Parameters
        self.alpha_I = params['alpha_I']
        self.alpha_H = params['alpha_H']
        self.Kp = params['Kp']
        self.Kav = params['Kav']
        self.Kv = params['Kv']
        self.KR = params['KR']
        self.tau_r = params['tau_r']
        self.Kh = params['Kh']
        self.Kc = params['Kc']
        self.Pc = params['Pc']
        self.V0 = params['V0']
        self.H0 = params['H0']
        self.Q0 = params['Q0']
        self.Pa0 = params['Pa0']
        self.K2 = params['K2']
        self.Gr = params['Gr']
        self.Gvu = params['Gvu']
        self.BIS0 = 93.4

        # Parameters used in equations
        self.Ka = self.Kv*self.Kav
        self.Kvu = self.K2 * (self.KR/self.tau_r)
        
        self.Pr = 1./self.tau_r
        self.Kr = self.KR/self.tau_r

        # Propofol Parameters
        self.Ce50 = params['Ce50']
        self.VD = params['VD']
        self.ke0 = params['ke0']
        self.k10 = params['k10']
        self.k12 = params['k12']
        self.k13 = params['k13']
        self.k21 = params['k21']
        self.k31 = params['k31']

        # Constants / Calculations
        self.Va0 = 0.3 * self.V0
        self.Vv0 = 0.7 * self.V0
        self.Pv0 = 6.0
        self.Vp0 = self.V0 - (self.H0 * self.V0)

        # Unstressed Volume
        self.Vu0 = self.Vv0 - (self.Pv0 / self.Kv)


    
    ## DYNAMICS
    def f(self, t, x, u):
        # States
        # - `Va, Vv` : arterial/venous volumes            **[L]**  
        # - `Vr`     : red-blood-cell volume              **[L]**
        # - `rF`     : interstitial fluid volume state    **[L]**
        # - `sR`     : TPR regulation state               **[mmHg·min/L]**
        # - `Q`      : cardiac output                     **[L/min]**  [NOTE THAT THE EQUATION FOR CARDIAC OUTPUT IS SLIGHTLY DIFFERENT FROM WHAT IS SHOWN IN THE JCMC PAPER.] 
        # - `sVU`    : venous unstressed-volume regulation state **[L]** 
        # - `m1,m2,m3`: propofol PK compartment masses           **[mg]**
        # - `Ce`     : propofol effect-site concentration        **[mg/L]**
        scaled_x = x / self.scaling
        Va, Vv, Vr, rF, sR, Q, sVU, m1, m2, m3, Ce = scaled_x.ravel() # Q is part of y
        # Inputs
        JI, JH, JP = u.ravel()
        # hct_add = hct_add[0]
        hct_add = 0 # for now, since it doesn't make sense to be nonzero atm

        # Constants / Calculations
        H = Vr / (Va + Vv) # part of y

        # Pressures
        # Pa = self.Pa0 + self.Ka * (Va - self.Va0) # part of y
        Pa = self.Pa0 - self.Ka * (self.Va0 - Va) # flip sign because interval shenanigans

        # Unstressed Volume
        Delta_Vu = (self.Kvu * sVU) + (self.Gvu * Ce) # part of y
        Vu = self.Vu0 + Delta_Vu
        # Vs = Vv - Vu
        Vs = -(Vu - Vv) # flip sign because interval shenanigans
        Pv = self.Kv * Vs # part of y
        sVU_dot = -self.Pr * sVU + (Pa - self.Pa0)

        # Cardiac Output
        # Q_dot = -self.Pc * (Q - self.Q0) + self.Kc * (Pv - self.Pv0)
        Q_dot = self.Pc * (self.Q0 - Q) - self.Kc * (self.Pv0 - Pv) # flip signs because interval shenanigans

        # Systemic Vascular Resistance
        R0 = (self.Pa0 - self.Pv0) / self.Q0
        # R = R0 - (self.Kr * sR) + self.Kh*(H - self.H0) - (self.Gr * Ce) # part of y
        R = R0 - (self.Kr * sR) - self.Kh*(self.H0 - H) - (self.Gr * Ce) #flip signs because interval shenanigans
        sR_dot = -self.Pr * sR + (Pa - self.Pa0)

        # Tissue Fluid Exchange
        # JF = self.Kp*(Va + Vv - self.Va0 - self.Vv0 - rF)
        JF = -self.Kp*(self.Va0 + self.Vv0 + rF - Va - Vv) # flip sign because interval shenanigans
        # rF_dot = ((1./(1+self.alpha_I))*JI - (1./(1+self.alpha_H))*JH)
        rF_dot = ((1./(1+self.alpha_I))*JI - (1./(1+self.alpha_H))*JH)
        # Blood Circulation
        # Va_dot = Q - ((Pa-Pv)/R) - JH - JF
        Va_dot = -(((Pa-Pv)/R) + JH + JF - Q) # flip sign because interval shenanigans
        # Vv_dot = -Q + ((Pa-Pv)/R) + JI
        Vv_dot = ((Pa-Pv)/R) + JI - Q # rearrange because interval shenanigans
        Vr_dot = -JH*H + JI*hct_add # why is hct_add not considered an input u?

        # PK and PD propofol
        m1_dot = (-(self.k10+self.k12+self.k13)*m1+self.k21*m2+self.k31*m3+JP)
        m2_dot = (self.k12*m1-self.k21*m2)
        m3_dot = (self.k13*m1-self.k31*m3)

        Vp = jnp.maximum(Va + Vv - Vr, 1e-9) # avoid divide-by-0
        Ce_dot = (-self.ke0*Ce + (self.Vp0*self.ke0*m1)/(self.VD*Vp))

        # gamma = jax.lax.cond(Ce <= self.Ce50, lambda x: 1.89, lambda x: 1.47, False) # these two lines are probably gonna cause some interval issues
        # BIS = self.BIS0*(1-jnp.pow(Ce, gamma)/(jnp.pow(self.Ce50, gamma)+jnp.pow(Ce, gamma))) # part of y

        final_outputs = [
            Va_dot,
            Vv_dot,
            Vr_dot,
            rF_dot,
            sR_dot,
            Q_dot,
            sVU_dot,
            m1_dot,
            m2_dot,
            m3_dot,
            Ce_dot
            ]
        
        return jnp.array(final_outputs) * self.scaling


# read parameter sets from csv
param_sets = []
with open('VP_Params_named.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for (i, row) in enumerate(reader):
        # print(i, row)
        if i == 0:
            # get parameter names
            param_names = row[:]
        else:
            # get parameter values and create param dict
            params_dict = {}
            for (j, name) in enumerate(param_names):
                if name == 'PatientType':
                    params_dict[name] = row[j]
                else:
                    params_dict[name] = float(row[j])
            # append to list
            param_sets.append(params_dict)
cur_params = param_sets[0] # choose param set

# read input sets from csv
inputs = []
with open('Inputs.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for (i, row) in enumerate(reader):
        # print(i, row)
        if i > 0:
            # append to list
            inputs.append([float(r) for r in row[1:]])
inputs = jnp.array(inputs)



_sys = HRPropofol(cur_params)
x_eq = jnp.array([
    cur_params["Va_ss"],
    cur_params["Vv_ss"],
    cur_params["Vr_ss"],
    0.0,
    0.0,
    cur_params["Q_ss"],
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
])
u_eq = jnp.array([0.0, 0.0, 0.0])

A_eq = jax.jacfwd(_sys.f, argnums=1)(0.0, x_eq, u_eq)
B_eq = jax.jacfwd(_sys.f, argnums=2)(0.0, x_eq, u_eq)
Q = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
R = jnp.eye(3)
K_eq, S_eq, _ = lqr(A_eq, B_eq, Q, R)
K_eq = -jnp.array(K_eq)
Th_eq = jnp.asarray(cholesky(S_eq, upper=True))


def ncm(x, ncm_net):
    # Killing field condition, no dependence in actuation directions
    Th_flat = ncm_net(x[:-4])
    Th = jnp.zeros((_sys.xlen, _sys.xlen), dtype=Th_flat.dtype)
    Th = Th.at[onp.triu_indices(_sys.xlen)].set(Th_flat)
    return Th + Th_eq


def control(x, control_net):
    return control_net(x) + K_eq @ (x - x_eq) + u_eq


if __name__ == "__main__":
    from neural_contraction.Propofol import CONTROLLER

    jnp.savez(
        CONTROLLER / "eq.npz", K_eq=K_eq, x_eq=x_eq, u_eq=u_eq, Th_eq=Th_eq, Q=Q, R=R
    )
