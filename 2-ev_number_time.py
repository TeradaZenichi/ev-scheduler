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
import random
# Set a seed for reproducibility
random.seed(42)

matplotlib.use('Agg')  # Use the 'Agg' backend for saving plots without needing Tkinte   
folder = "data"


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

πs = {}
for s in Ωs:
    πs[s] = fs[s]['prob']

Ωc = [0] + data['off-grid']['set']
πc = {}
for c in Ωc:
    if c == 0:
        πc[c] = 1 - data['off-grid']['prob']
    else:
        πc[c] = data['off-grid']['prob'] / len(data['off-grid']['set'])

GHG_LIM = 1e8
EV_IDLE_LIM = 1e8

def generate_random_profile():
    arrival_minutes = random.randint(0, 23 * 60 // Δt) * Δt
    departure_minutes = random.randint(0, 23 * 60 // Δt) * Δt
    arrival = (datetime.strptime('00:00', "%H:%M") + timedelta(minutes=arrival_minutes)).strftime("%H:%M")
    departure = (datetime.strptime('00:00', "%H:%M") + timedelta(minutes=departure_minutes)).strftime("%H:%M")
    return {
        "arrival": arrival,
        "departure": departure,
        "SoCini": round(random.uniform(0.2, 0.8), 2),
        "Emax": 10 * random.randint(3, 8),
        "eff": round(random.uniform(0.85, 0.99), 2),
        "Pmax_c": 10 * random.randint(3, 5),
        "Pmax_d": 10 * random.randint(0, 3)
    }


for i in range(0, 100):
    
    if i > 0:
        new_id = str(len(EV) + 1)
        EV[new_id] = generate_random_profile()

    #define probabilities
    


    start = time.time()

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
            total_opex += model.OPEX * ((1 + data["OPEX"]["rate"]) ** (n - 1))

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
        evcs_oem = data["EVCS"]["O&M"] * model.NEVCS * data["EVCS"]["CAPEX"]
        return model.OPEX == eds_opex + bess_oem + ev_opex + tg_opex  + pv_oem + evcs_oem
    model.opex_constraint = Constraint(rule=opex_constraint_rule)

    # Update power balance constraint
    def power_balance_rule(model, t, c, s):
        pv = model.PPVmax * fs[s]['pv'][t]  # PV generation
        load = data["EDS"]["LOAD"] * fs[s]['load'][t]  # Demand
        return (
            model.PEDS[t, c, s] + pv + sum(model.PEV_d[ev, t, c, s] for ev in Ωev) + model.PBESS_d[t, c, s] + model.PTG[t, c, s] ==
            sum(model.PEV_c[ev, t, c, s] for ev in Ωev) + model.PBESS_c[t, c, s] + load
        )
    model.power_balance = Constraint(Ωt, Ωc, Ωs, rule=power_balance_rule)

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
        return model.EBESS[t2, c, s] == model.EBESS[t1, c, s] + ηBESS * model.PBESS_c[t2, c, s] * (Δt / 60) - model.PBESS_d[t2, c, s] * (ηBESS * Δt / 60)
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
            return model.EEV[ev, t2, c, s] == model.EEV[ev, t1, c, s] + ηEV * model.PEV_c[ev, t2, c, s] * (Δt / 60) - model.PEV_d[ev, t2, c, s] * (ηEV * Δt / 60)
        elif ta > td and not (t0 > td and t0 <= ta):
            return model.EEV[ev, t2, c, s] == model.EEV[ev, t1, c, s] + ηEV * model.PEV_c[ev, t2, c, s] * (Δt / 60) - model.PEV_d[ev, t2, c, s] * (ηEV * Δt / 60)
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



    results = SolverFactory('gurobi').solve(model)

    end = time.time()

    folder = 'Results_pareto'

    # import json file if exists
    if os.path.exists(f'{folder}/ev_number_results.json'):
        with open(f'{folder}/ev_number_results.json') as file:
            results = json.load(file)
    else:
        results = {}

    results[len(EV.keys())] = {
        "Total Time [s]": end - start,
        "Total Cost [MUSD]": value(model.objective)/1e6,
        "PPVmax [kW]": model.PPVmax.value,
        "EmaxBESS [kWh]": model.EmaxBESS.value,
        "TG_MAX_CAP [kW]": model.TG_MAX_CAP.value,
        "NEVCS": model.NEVCS.value,
        "GHG [kgCO2]": model.GHG.value,
        "EV_IDLE_LIM": EV_IDLE_LIM,
    }

    with open(f'{folder}/ev_number_results.json', 'w') as file:
        json.dump(results, file, indent=4)

    with open(f'{folder}/EVs.json', 'w') as file:
        json.dump(EV, file, indent=4)