from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import matplotlib.pyplot as plt  # Importing pyplot as plt
from pyomo.environ import *
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import time
import json
import re
import os

matplotlib.use('Agg')  # Use the 'Agg' backend for saving plots without needing Tkinte
import matplotlib as mpl
mpl.rc('font', family='serif', serif='cmr10')
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams.update({'font.size':14})
plt.rcParams["axes.formatter.use_mathtext"] = True


folder = "data"

start = time.time()

# Load the JSON file
with open(f'{folder}/cost.json') as file:
    cost = json.load(file)

with open(f'{folder}/scenarios.json') as file:
    fs = json.load(file)

with open(f'{folder}/parameters.json') as file:
    data = json.load(file)
    
with open(f'{folder}/EVs.json') as file:
    EV = json.load(file)

# timestep
Δt = data['timestep']

# Set definition
Ωev = EV.keys()
Ωs  = data['scenarios']


Ωt  = []
t = datetime.strptime('00:00', "%H:%M")
while t < datetime.strptime('23:59', "%H:%M"):
    Ωt.append(t.strftime("%H:%M"))
    t += timedelta(minutes=Δt)


Ωc = [0] + data['off-grid']['set']  

#define probabilities
πc = {}
for c in Ωc:
    if c == 0:
        πc[c] = 1 - data['off-grid']['prob']
    else:
        πc[c] = data['off-grid']['prob'] / len(data['off-grid']['set'])

πs = {}
for s in Ωs:
    πs[s] = fs[s]['prob']


GHG_LIM = 9912.413133647737
EV_IDLE_LIM = 150


model = ConcreteModel()

# Variable definition with contingency scenarios
model.PEDS = Var(Ωt, Ωc, Ωs, domain=Reals, bounds=(data["EDS"]['Pmin'], data["EDS"]['Pmax']))  # Substation power
model.PEDSp = Var(Ωt, Ωc, Ωs, domain=NonNegativeReals)  # Energy bought from the grid
model.PEDSn = Var(Ωt, Ωc, Ωs, domain=NonNegativeReals)  # Energy sold to the grid

# BESS variables
model.PBESS_c = Var(Ωt, Ωc, Ωs, within=NonNegativeReals)  # BESS charging power
model.PBESS_d = Var(Ωt, Ωc, Ωs, within=NonNegativeReals)  # BESS discharging power
model.EBESS = Var(Ωt, Ωc, Ωs, within=NonNegativeReals)  # BESS State of Charge
model.EmaxBESS = Var(within=NonNegativeReals)  # BESS maximum energy capacity

# EV operation variables
model.EEV = Var(Ωev, Ωt, Ωc, Ωs, domain=NonNegativeReals)  # EV State of Charge
model.PEV_c = Var(Ωev, Ωt, Ωc, Ωs, domain=NonNegativeReals)  # EV charging power
model.PEV_d = Var(Ωev, Ωt, Ωc, Ωs, domain=NonNegativeReals)  # EV discharging power
model.αEV = Var(Ωev, Ωt, Ωc, Ωs, domain=Binary)  # EV charging factor

# Thermal generator variables
model.TG_MAX_CAP = Var(within=NonNegativeReals)  # Maximum installed capacity (affects CAPEX)
model.PTG = Var(Ωt, Ωc, Ωs, domain=NonNegativeReals)  # Power output (affects OPEX)

# PV variable
model.PPVmax = Var(domain=NonNegativeReals)  # Maximum PV generation
model.PPV = Var(Ωt, Ωc, Ωs, domain=NonNegativeReals)  # PV generation
model.XPV = Var(Ωt, Ωc, Ωs, domain=Binary)  # PV shedding

# LOAD variable
model.PLOAD = Var(Ωt, Ωc, Ωs, domain=NonNegativeReals)  # Load
model.XLOAD = Var(Ωt, Ωc, Ωs, domain=Binary)  # Load shedding

# Economic variables
model.CAPEX = Var(within=NonNegativeReals)  # Total CAPEX
model.OPEX = Var(within=NonNegativeReals)  # Total OPEX

# Number of charging stations
model.NEVCS = Var(domain=NonNegativeReals)

# Multi objective variables
model.EVidle = Var(Ωev, Ωc, Ωs, within=NonNegativeReals)  # Total EV idle time
model.GHG = Var(within=NonNegativeReals)  # Total GHG emissions

def objective_rule(model):
    total_opex = 0
    for n in range(1, data["OPEX"]["years"] + 1):
        total_opex += model.OPEX / ((1 + data["OPEX"]["rate"]) ** (n))

    return total_opex + model.CAPEX
model.objective = Objective(rule=objective_rule, sense=minimize)


def capex_constraint_rule(model):
    PV          = model.PPVmax * data["PV"]['CAPEX']  # PV installation cost
    BESS_CAPEX  = model.EmaxBESS * data["BESS"]["CAPEX"]
    TG          = model.TG_MAX_CAP * data["TG"]["CAPEX"]
    NEVCS       = model.NEVCS * data["EVCS"]["CAPEX"]
    return model.CAPEX == PV + BESS_CAPEX + TG + NEVCS
model.capex_constraint = Constraint(rule=capex_constraint_rule)

def opex_constraint_rule(model):
    eds_opex = 365 * sum(πc[c] * πs[s] * (cost[t] * model.PEDSp[t, c, s] * (Δt / 60)) for t in Ωt for c in Ωc for s in Ωs)
    
    ev_opex = 365 * sum(πc[c] * πs[s] * (
        data["EVCS"]["discharging_cost"] * model.PEV_d[ev, t, c, s] * (Δt / 60))
        for ev in Ωev for t in Ωt for c in Ωc for s in Ωs
    )
    tg_opex = 365 * sum(πc[c] * πs[s] * (
        data["TG"]["OPEX"] * model.PTG[t, c, s] * (Δt / 60))
        for t in Ωt for c in Ωc for s in Ωs
    )
    pv_oem    = data['PV']['O&M'] * model.PPVmax * data["PV"]['CAPEX']
    bess_oem = 365 * sum(πc[c] * πs[s] * (
        data["BESS"]["charging_cost"] * model.PBESS_c[t, c, s] * (Δt / 60) +
        data["BESS"]["discharging_cost"] * model.PBESS_d[t, c, s] * (Δt / 60))
        for t in Ωt for c in Ωc for s in Ωs
    )
    
    load_shedding = 365 * sum(πc[c] * πs[s] * (
        data["LOAD"]["cost"] * model.XLOAD[t, c, s] * data["LOAD"]["Pmax"] * fs[s]['load'][t] * (Δt / 60))
        for t in Ωt for c in Ωc for s in Ωs
    )
    evcs_oem = data["EVCS"]["O&M"] * model.NEVCS * data["EVCS"]["CAPEX"]
    return model.OPEX == eds_opex + bess_oem + ev_opex + tg_opex  + pv_oem + evcs_oem + load_shedding
model.opex_constraint = Constraint(rule=opex_constraint_rule)

# Update power balance constraint
def power_balance_rule(model, t, c, s):
    return (
        model.PEDS[t, c, s] + model.PPV[t, c, s] + sum(model.PEV_d[ev, t, c, s] for ev in Ωev) + model.PBESS_d[t, c, s] + model.PTG[t, c, s] ==
        sum(model.PEV_c[ev, t, c, s] for ev in Ωev) + model.PBESS_c[t, c, s] + model.PLOAD[t, c, s]
    )
model.power_balance = Constraint(Ωt, Ωc, Ωs, rule=power_balance_rule)


################################################################################
################################## PV constraints ##############################
################################################################################

def pv_limit_rule_1(model, t, c, s):
    return model.PPV[t, c, s] <= 1e6 * (1 - model.XPV[t, c, s])
model.pv_limit_1 = Constraint(Ωt, Ωc, Ωs, rule=pv_limit_rule_1)

def pv_limit_rule_2(model, t, c, s):
    return model.PPV[t, c, s] <= model.PPVmax * fs[s]['pv'][t]
model.pv_limit_2 = Constraint(Ωt, Ωc, Ωs, rule=pv_limit_rule_2)

def pv_limit_rule_3(model, t, c, s):
    return model.PPV[t, c, s] >= model.PPVmax * fs[s]['pv'][t] - 1e6 * model.XPV[t, c, s]
model.pv_limit_3 = Constraint(Ωt, Ωc, Ωs, rule=pv_limit_rule_3)

################################################################################
################################## LOAD constraints ############################
################################################################################

def load_shedding_rule(model, t, c, s):
    return model.PLOAD[t, c, s] == data["LOAD"]["Pmax"] * fs[s]['load'][t] * (1 - model.XLOAD[t, c, s])
model.load_shedding = Constraint(Ωt, Ωc, Ωs, rule=load_shedding_rule)


################################################################################
##################### Multi objective constraints ##############################
################################################################################

def EV_idle_constraint_rule(model, ev, c, s):
    # Convertendo os horários de chegada e saída para datetime
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    
    if ta > td:
        td += timedelta(days=1)  # Adicionamos um dia a td

    idle_periods = (td - ta).total_seconds() / 60 / Δt  # Número de intervalos de tempo
    return idle_periods - sum(model.αEV[ev, t, c, s] for t in Ωt) + 1 == model.EVidle[ev, c, s]
model.EV_idle_constraint = Constraint(Ωev, Ωc, Ωs, rule=EV_idle_constraint_rule)


def anual_GHG_constraint_rule(model):
    eds_emission = 365 * sum(πc[c] * πs[s] * (
        data["EDS"]["GHG"] * model.PEDSp[t, c, s] * (Δt / 60))
        for t in Ωt for c in Ωc for s in Ωs
    )
    tg_emission = 365 * sum(πc[c] * πs[s] * (
        data["TG"]["GHG"] * model.PTG[t, c, s] * (Δt / 60))
        for t in Ωt for c in Ωc for s in Ωs
    )
    return model.GHG == eds_emission + tg_emission
model.anual_GHG_constraint = Constraint(rule=anual_GHG_constraint_rule)


def EV_idle_limit_constraint_rule(model, ev, c, s):
    limit = EV_IDLE_LIM / Δt
    return model.EVidle[ev, c, s] <= limit
model.EV_idle_limit_constraint = Constraint(Ωev, Ωc, Ωs, rule=EV_idle_limit_constraint_rule)


def GHG_limit_constraint_rule(model):
    return model.GHG <= GHG_LIM
model.GHG_limit_constraint = Constraint(rule=GHG_limit_constraint_rule)


################################################################################
############################# TG constraints ###################################
################################################################################

def tg_power_rule(model, t, c, s):
    return model.PTG[t, c, s] <= model.TG_MAX_CAP  # Power output cannot exceed maximum installed capacity
model.tg_power_constraint = Constraint(Ωt, Ωc, Ωs, rule=tg_power_rule)


################################################################################
############################ EDS constraints ###################################
################################################################################
def eds_contingency_rule(model, t, c, s):
    if c == 0:
        return Constraint.Skip
    
    tt = datetime.strptime(t, "%H:%M")
    tc = datetime.strptime(c, "%H:%M")
    if tc <= tt < tc + timedelta(hours=data['off-grid']['duration']):
        return model.PEDS[t, c, s] == 0  # Substation power is zero during contingency
    else:
        return Constraint.Skip  # No constraint outside contingency periods
model.eds_contingency = Constraint(Ωt, Ωc, Ωs, rule=eds_contingency_rule)

def eds_power_rule(model, t, c, s):
    return model.PEDS[t, c, s] == model.PEDSp[t, c, s] - model.PEDSn[t, c, s]
model.eds_power_constraint = Constraint(Ωt, Ωc, Ωs, rule=eds_power_rule)


################################################################################
#########################   BESS constraints ###################################
################################################################################

ηBESS = data["BESS"]["efficiency"]

def bess_energy_rule(model, t, c, s):
    return model.EBESS[t, c, s] <= model.EmaxBESS
model.bess_energy_constraint = Constraint(Ωt, Ωc, Ωs, rule=bess_energy_rule)

def bess_power_rule_c(model, t, c, s):
    return model.PBESS_c[t, c, s] <= model.EmaxBESS * data["BESS"]['crate']
model.bess_power_c = Constraint(Ωt, Ωc, Ωs, rule=bess_power_rule_c)

def bess_power_rule_d(model, t, c, s):
    return model.PBESS_d[t, c, s] <= model.EmaxBESS * data["BESS"]['crate']
model.bess_power_d = Constraint(Ωt, Ωc, Ωs, rule=bess_power_rule_d)

def bess_energy_update_rule(model, t, c, s):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    return model.EBESS[t2, c, s] == model.EBESS[t1, c, s] + ηBESS * model.PBESS_c[t2, c, s] * (Δt / 60) - model.PBESS_d[t2, c, s] * (Δt /(ηBESS * 60))
model.bess_energy_update = Constraint(Ωt, Ωc, Ωs, rule=bess_energy_update_rule)

def bess_charging_linearization_rule(model, t, c, s):
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    return model.PBESS_c[t2, c, s] <= (model.EmaxBESS - model.EBESS[t1, c, s]) / (ηBESS * Δt / 60)
model.bess_charging_linearization = Constraint(Ωt, Ωc, Ωs, rule=bess_charging_linearization_rule)

def bess_discharging_linearization_1_rule(model, t, c, s):
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    return model.PBESS_d[t2, c, s] <= model.EBESS[t1, c, s] * ηBESS / (Δt / 60)
model.bess_discharging_linearization_1 = Constraint(Ωt, Ωc, Ωs, rule=bess_discharging_linearization_1_rule)

def bess_discharging_linearization_2_rule(model, t, c, s):
    return model.PBESS_d[t, c, s] <= model.EmaxBESS * data["BESS"]['crate'] - model.PBESS_c[t, c, s]
model.bess_discharging_linearization_2 = Constraint(Ωt, Ωc, Ωs, rule=bess_discharging_linearization_2_rule)

def bess_initial_energy_rule(model, c, s):
    return model.EBESS[Ωt[0], c, s] == data["BESS"]["SoCini"] * model.EmaxBESS
model.bess_initial_energy = Constraint(Ωc, Ωs, rule=bess_initial_energy_rule)

def bess_offgrid_rule(model, t, c, s):
    if c != 0:
        tt = datetime.strptime(t, "%H:%M")
        tc = datetime.strptime(c, "%H:%M")
        if tt < tc:
            return model.EBESS[t, c, s] == model.EBESS[t, 0, s]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bess_offgrid = Constraint(Ωt, Ωc, Ωs, rule=bess_offgrid_rule)

def bess_initial_final_energy_rule(model, c, s):
    return model.EBESS[Ωt[-1], c, s] == data["BESS"]["SoCini"] * model.EmaxBESS
model.bess_initial_final_energy = Constraint(Ωc, Ωs, rule=bess_initial_final_energy_rule)

################################################################################
#########################   ALPHA constraints ###################################
################################################################################
def alpha_ev_update(model, ev, t, c, s):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")
    if ta < td and t0 > ta and t0 <= td:
        return model.αEV[ev, t, c, s] <= 1
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.αEV[ev, t, c, s] <= 1
    elif t0 == ta:
        return model.αEV[ev, t, c, s] <= 1
    else:
        return model.αEV[ev, t, c, s] <= 0
model.alpha_ev_update = Constraint(Ωev, Ωt, Ωc, Ωs, rule=alpha_ev_update)


def ev_charging_power_rule(model, ev, t, c, s):
    return model.PEV_c[ev, t, c, s] <= model.αEV[ev, t, c, s] * EV[ev]['Pmax_c']
model.ev_charging_power_alpha = Constraint(Ωev, Ωt, Ωc, Ωs, rule=ev_charging_power_rule)

def ev_discharging_power_rule(model, ev, t, c, s):
    return model.PEV_d[ev, t, c, s] <= model.αEV[ev, t, c, s] * EV[ev]['Pmax_d']
model.ev_discharging_power_alpha = Constraint(Ωev, Ωt, Ωc, Ωs, rule=ev_discharging_power_rule)

def ev_connection_constraint_rule(model, ev, t, c, s):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    if ta < td and t0 > ta and t0 <= td:
        return model.αEV[ev, t2, c, s] >= model.αEV[ev, t1, c, s]
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.αEV[ev, t2, c, s] >= model.αEV[ev, t1, c, s]
    else:
        return Constraint.Skip
model.ev_connection_constraint = Constraint(Ωev, Ωt, Ωc, Ωs, rule=ev_connection_constraint_rule)


# EVCS constraints
def evcs_constraint_rule(model, t, c, s):
    return model.NEVCS >= sum(model.αEV[ev, t, c, s] for ev in Ωev)
model.evcs_constraint = Constraint(Ωt, Ωc, Ωs, rule=evcs_constraint_rule)

################################################################################
###########################   EV constraints ###################################
################################################################################

def ev_energy_update_rule(model, ev, t, c, s):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    ηEV = EV[ev]['eff']
    if ta < td and t0 > ta and t0 <= td:
        return model.EEV[ev, t2, c, s] == model.EEV[ev, t1, c, s] + ηEV * model.PEV_c[ev, t2, c, s] * (Δt / 60) - model.PEV_d[ev, t2, c, s] * (Δt / (ηEV * 60))
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.EEV[ev, t2, c, s] == model.EEV[ev, t1, c, s] + ηEV * model.PEV_c[ev, t2, c, s] * (Δt / 60) - model.PEV_d[ev, t2, c, s] * (Δt / (ηEV * 60))
    elif t0 == ta:
        return model.EEV[ev, t, c, s] == EV[ev]['SoCini'] * EV[ev]['Emax']
    else:
        return model.EEV[ev, t, c, s] == 0
model.ev_energy_update = Constraint(Ωev, Ωt, Ωc, Ωs, rule=ev_energy_update_rule)


def ev_energy_offgrid_rule(model, ev, t, c, s):
    if c != 0:
        tt = datetime.strptime(t, "%H:%M")
        tc = datetime.strptime(c, "%H:%M")
        if tt < tc:
            return model.EEV[ev, t, c, s] == model.EEV[ev, t, 0, s]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.ev_energy_offgrid = Constraint(Ωev, Ωt, Ωc, Ωs, rule=ev_energy_offgrid_rule)

# Modified version of the V2G constraints that handles time correctly
def v2g_constraints(model, ev, t, c, s):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")  
    ηEV = EV[ev]['eff']    
    return model.PEV_c[ev, t2, c, s] <= (EV[ev]['Emax'] - model.EEV[ev, t1, c, s])  / (ηEV * Δt / 60) 

model.v2g_charge = Constraint(Ωev, Ωt, Ωc, Ωs, rule=v2g_constraints)

def v2g_discharge_constraints(model, ev, t, c, s):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    ηEV = EV[ev]['eff']
    return model.PEV_d[ev, t2, c, s] <= (model.EEV[ev, t1, c, s] * ηEV) / (Δt / 60)
model.v2g_discharge = Constraint(Ωev, Ωt, Ωc, Ωs, rule=v2g_discharge_constraints)

def v2g_discharge_max(model, ev, t, c, s):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    # Discharge power should not exceed the calculated max value
    return model.PEV_d[ev, t2, c, s] <= EV[ev]['Pmax_d'] - (EV[ev]['Pmax_d'] / EV[ev]['Pmax_c']) * model.PEV_c[ev, t2, c, s]
model.v2g_discharge_max = Constraint(Ωev, Ωt, Ωc, Ωs, rule=v2g_discharge_max)


def EV_departure_rule(model, ev, c, s):
    return model.EEV[ev, EV[ev]['departure'], c, s] == EV[ev]['Emax']
model.EV_departure = Constraint(Ωev, Ωc, Ωs, rule=EV_departure_rule)

results = SolverFactory('gurobi').solve(model, options={'MIPGap': 0.1})

end = time.time()

contingency_results = {}
for c in Ωc:
    for s in Ωs:
        contingency_results[(c,s)] = {
            'EDS': [model.PEDS[t, c, s].value for t in Ωt],
            'PV': [model.PPV[t, c, s].value for t in Ωt],
            'PLOAD': [model.PLOAD[t, c, s].value for t in Ωt],
            'PBESS_c': [model.PBESS_c[t, c, s].value for t in Ωt],
            'PBESS_d': [model.PBESS_d[t, c, s].value for t in Ωt],
            'PEV_c': [sum(model.PEV_c[ev, t, c, s].value for ev in Ωev) for t in Ωt],
            'PEV_d': [sum(model.PEV_d[ev, t, c, s].value for ev in Ωev) for t in Ωt],
            'PV': [model.PPVmax.value * fs[s]['pv'][t] for t in Ωt],
            'Demand': [data["LOAD"]["Pmax"] * fs[s]['load'][t] for t in Ωt],
            'PTG': [model.PTG[t, c, s].value for t in Ωt]  # Gerador térmico
        }

# ==========================================
# Verificar status do solver e imprimir resultados
# ==========================================
print(f"Tempo de execução: {end - start:.2f} seconds")
print("Resultados de Otimização:")
if results.solver.status == SolverStatus.ok:
    print(f"Total Cost: {value(model.objective):.2f} USD")
    print(f"Maximum Solar Generation Capacity (PPVmax): {model.PPVmax.value:.2f} kW")
    print(f"Maximum BESS Energy Capacity (EmaxBESS): {model.EmaxBESS.value:.2f} kWh")
    print(f"Maximum Thermal Generator Capacity (TG_MAX_CAP): {model.TG_MAX_CAP.value:.2f} kW")
    print(f"Number of EV Charging Stations (NEVCS): {model.NEVCS.value:.0f}")
    print(f"Anual GHG Emissions: {model.GHG.value:.2f} kgCO2")
    print(f"Total energy by V2G: {sum(sum( πs[s] * πc[c] * model.PEV_d[ev, t, c, s].value for t in Ωt) for ev in Ωev for c in Ωc for s in Ωs):.2f} kWh")
    print(f"Total Load shedding cost: {sum(sum( πs[s] * πc[c] * model.XLOAD[t, c, s].value * data['LOAD']['cost'] * data['LOAD']['Pmax'] * fs[s]['load'][t] * (Δt / 60) for t in Ωt) for c in Ωc for s in Ωs):.2f} USD")
    print(f"Total PV shedding: {sum(sum( πs[s] * πc[c] * model.XPV[t, c, s].value * 1e6 for t in Ωt) for c in Ωc for s in Ωs):.2f} kWh")
else:
    print("Solution not found")


idle_time_data = []
for ev in EV.keys():
    for s in Ωs:
        for c in Ωc:
            idle_time = model.EVidle[ev, c, s].value  # Assumindo que já foi resolvido
            idle_time_data.append([ev, s, c, idle_time])

df_idle_time = pd.DataFrame(idle_time_data, columns=["ev", "s", "c", "idle_time"])


# Garante que a pasta 'Results' exista (para evitar erro ao salvar)
os.makedirs("Results", exist_ok=True)

# Save results to txt
with open("Results/1-Results.txt", "w") as f:
    f.write(f"Total Cost: {value(model.objective):.2f} USD\n")
    f.write(f"Maximum Solar Generation Capacity (PPVmax): {model.PPVmax.value:.2f} kW\n")
    f.write(f"Maximum BESS Energy Capacity (EmaxBESS): {model.EmaxBESS.value:.2f} kWh\n")
    f.write(f"Maximum Thermal Generator Capacity (TG_MAX_CAP): {model.TG_MAX_CAP.value:.2f} kW\n")
    f.write(f"Number of EV Charging Stations (NEVCS): {model.NEVCS.value:.0f}\n")
    f.write(f"Anual GHG Emissions: {model.GHG.value:.2f} kgCO2\n")
    

    f.write("\nIdle time for each EV:\n")
    f.write(df_idle_time.to_string(index=False))


# ==========================================
# 1) PLOTS DE CADA CENÁRIO DE CONTINGÊNCIA
# ==========================================
colors = {
    'Substation': 'blue',
    'Demand': 'black',
    'Solar': 'orange',
    'Total EV charging Power': 'purple',
    'Total EV discharging Power': 'pink',
    'Net BESS Power': 'green',
    'Thermal Generator': 'red'
}



for c in Ωc:
    for s in Ωs:
        # Garante que c seja string segura para nome de arquivo
        safe_c = str(c)
        safe_c = re.sub(r'[^a-zA-Z0-9_]+', '_', safe_c)
        safe_s = str(s)
        safe_s = re.sub(r'[^a-zA-Z0-9_]+', '_', safe_s)

        EDS_values = contingency_results[(c,s)]['EDS']
        PLOAD_values = contingency_results[(c,s)]['PLOAD']
        PPV_values = contingency_results[(c,s)]['PV']
        PBESS_c_values = contingency_results[(c,s)]['PBESS_c']
        PBESS_d_values = contingency_results[(c,s)]['PBESS_d']
        PEV_c_values = contingency_results[(c,s)]['PEV_c']
        PEV_d_values = contingency_results[(c,s)]['PEV_d']
        PTG_values = contingency_results[(c,s)]['PTG']
        
        net_BESS_power = [PBESS_c_values[t_idx] - PBESS_d_values[t_idx] for t_idx in range(len(Ωt))]
        discharging = [-1*PBESS_d_values[t_idx] for t_idx in range(len(Ωt))]
        PTG_values = [-1 * PTG_values[t_idx] for t_idx in range(len(Ωt))]
        PEV_d_values = [-1 * PEV_d_values[t_idx] for t_idx in range(len(Ωt))]

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        ax.plot(Ωt, EDS_values, label='EDS Power', color=colors['Substation'], linewidth=2)
        ax.plot(Ωt, PLOAD_values, label='Demand', color=colors['Demand'], linewidth=2, linestyle='--', marker='o', markersize=1)
        if model.PPVmax.value > 0:
            ax.plot(Ωt, PPV_values, label='Solar Generation', color=colors['Solar'], linewidth=2)
        ax.bar(Ωt, PEV_c_values, label='Total EV charging Power', color=colors['Total EV charging Power'], linewidth=2)
        ax.bar(Ωt, PEV_d_values, label='Total EV discharging Power', color=colors['Total EV discharging Power'], linewidth=2)
        if model.TG_MAX_CAP.value > 0:
            ax.plot(Ωt, PTG_values, label='Thermal Generator Power', color=colors['Thermal Generator'], linewidth=2)
        if model.EmaxBESS.value > 0:
            ax.bar(Ωt, net_BESS_power, color=colors['Net BESS Power'], alpha=0.5,
                label='Net BESS Power (BESSc - BESSd)')
        ax.set_title(f"Operational Power - Contingency {c} - Scenario {s}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Timestamp", fontsize=12)
        ax.set_ylabel("Power (kW)", fontsize=12)
        ax.grid(True, which='both', linestyle=':', color='lightgray')
        ax.legend(loc='upper left', fontsize=10, frameon=True)

        num_ticks = 10  
        tick_positions = list(range(0, len(Ωt), max(1, len(Ωt) // num_ticks)))  
        tick_labels = [Ωt[i] for i in tick_positions] 

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')  
        ax.set_ylim(-200, 200)
        plt.tight_layout()
        plt.savefig(f"Results/Contingency_{safe_c}_Scenario_{safe_s}_Power_Graph.pdf", dpi=300)
        plt.close()


# ==========================================
# 2) SoC HEATMAP
# ==========================================
def to_time(hhmm):
    return datetime.strptime(hhmm, "%H:%M").time()

def is_ev_present(t: datetime.time, arrival: datetime.time, departure: datetime.time) -> bool:
    if arrival < departure:
        return arrival <= t <= departure
    else:
        return not (departure < t < arrival)  



colorscale = [
    (0.0, 'lightyellow'),  # SoC=0
    (0.5, 'orange'),       # SoC=0.5
    (1.0, 'darkred')       # SoC=1
]
cmap_custom = LinearSegmentedColormap.from_list('CustomSoC', colorscale)

for c in Ωc:
    for s in Ωs:
        safe_c = re.sub(r'[^a-zA-Z0-9_]+', '_', str(c))
        safe_s = re.sub(r'[^a-zA-Z0-9_]+', '_', str(s))

        ev_soc_values = {
            ev: [model.EEV[ev, t, c, s].value / EV[str(ev)]['Emax'] for t in Ωt]
            for ev in Ωev
        }
        df_ev_soc = pd.DataFrame(ev_soc_values, index=Ωt)

        alpha_ev_values = {
            ev: [model.αEV[ev, t, c, s].value for t in Ωt]
            for ev in Ωev
        }
        df_ev_alpha = pd.DataFrame(alpha_ev_values, index=Ωt)

        if model.EmaxBESS.value > 0:
            bess_soc_values = {
                'BESS': [model.EBESS[t, c, s].value / model.EmaxBESS.value for t in Ωt]
            }
            df_bess_soc = pd.DataFrame(bess_soc_values, index=Ωt)
        else:
            df_bess_soc = pd.DataFrame(0, index=Ωt, columns=['BESS'])

        df_combined = pd.concat([df_ev_soc, df_bess_soc], axis=1)
        df_annotations = pd.DataFrame(index=df_combined.index, columns=df_combined.columns, dtype=object)
        times_as_time = [to_time(tt) for tt in Ωt]
        arrival_time = {}
        departure_time = {}
        for ev in Ωev:
            arr_str = EV[str(ev)]['arrival']
            dep_str = EV[str(ev)]['departure']
            arrival_time[ev] = to_time(arr_str)
            departure_time[ev] = to_time(dep_str)

        for row_idx, t_str in enumerate(Ωt):
            t_time = times_as_time[row_idx]
            for col in df_combined.columns:
                soc_val = df_combined.loc[t_str, col]  # SoC
                
                if col == 'BESS':
                    # df_annotations.loc[t_str, col] = f"{soc_val*100:.1f}%"
                    df_annotations.loc[t_str, col] = f"{soc_val:.2f}"
                else:
                    alpha_val = df_ev_alpha.loc[t_str, col]
                    arr_t = arrival_time[col]
                    dep_t = departure_time[col]

                    if is_ev_present(t_time, arr_t, dep_t):
                        # df_annotations.loc[t_str, col] = f"{soc_val*100:.1f}% - {int(alpha_val)}"
                        df_annotations.loc[t_str, col] = f"{int(alpha_val)}"
                    else:
                        df_annotations.loc[t_str, col] = ""

        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        sns.heatmap(
            df_combined.T,
            cmap=cmap_custom,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            cbar_kws={'label': 'State of Charge (SoC)'},
            annot=df_annotations.T,
            fmt="",
            annot_kws={'size': 5, 'rotation': 0},  # Fonte menor e rotação
            ax=ax
        )

        ax.set_title(f"EV & BESS SoC | Connection Status - Contingency {c} Scenario {s}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Timestamp", fontsize=12)
        ax.set_ylabel("EVs and BESS", fontsize=12)
        num_ticks = 10
        tick_positions = list(range(0, len(Ωt), max(1, len(Ωt) // num_ticks))) 
        tick_labels = [Ωt[i] for i in tick_positions]  

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')  # Rotaciona para melhor leitura

        plt.tight_layout()
        plt.savefig(f"Results/SoC_and_alphaEV_Heatmap_c{safe_c}_s{safe_s}.pdf", dpi=300)
        plt.close()