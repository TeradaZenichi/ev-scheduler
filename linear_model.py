from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import matplotlib.pyplot as plt  # Importing pyplot as plt
from pyomo.environ import *
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import json
import re
import os

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


Ωc = [0] + data['off-grid']['set']  

#define probabilities
πc = {}
for c in Ωc:
    if c == 0:
        πc[c] = 1 - data['off-grid']['prob']
    else:
        πc[c] = data['off-grid']['prob'] / len(data['off-grid']['set'])


# Definindo o modelo
model = ConcreteModel()


# Variable definition with contingency scenarios
model.PS = Var(Ωt, Ωc, domain=Reals, bounds=(data["EDS"]['Pmin'], data["EDS"]['Pmax']))  # Substation power
model.PSp = Var(Ωt, Ωc, domain=NonNegativeReals)  # Energy bought from the grid
model.PSn = Var(Ωt, Ωc, domain=NonNegativeReals)  # Energy sold to the grid

# BESS variables
model.PBESS_c = Var(Ωt, Ωc, within=NonNegativeReals)  # BESS charging power
model.PBESS_d = Var(Ωt, Ωc, within=NonNegativeReals)  # BESS discharging power
model.EBESS = Var(Ωt, Ωc, within=NonNegativeReals)  # BESS State of Charge
model.EmaxBESS = Var(within=NonNegativeReals)  # BESS maximum energy capacity

# EV operation variables
model.EEV = Var(Ωev, Ωt, Ωc, domain=NonNegativeReals)  # EV State of Charge
model.PEV_c = Var(Ωev, Ωt, Ωc, domain=NonNegativeReals)  # EV charging power
model.PEV_d = Var(Ωev, Ωt, Ωc, domain=NonNegativeReals)  # EV discharging power
model.αEV = Var(Ωev, Ωt, Ωc, domain=Binary)  # EV charging factor

# Thermal generator variables
model.TG_MAX_CAP = Var(within=NonNegativeReals)  # Maximum installed capacity (affects CAPEX)
model.PTG = Var(Ωt, Ωc, domain=NonNegativeReals)  # Power output (affects OPEX)

# PV variable
model.PPVmax = Var(domain=NonNegativeReals)  # Maximum PV generation

# Economic variables
model.CAPEX = Var(within=NonNegativeReals)  # Total CAPEX
model.OPEX = Var(within=NonNegativeReals)  # Total OPEX

# Number of charging stations
model.NEVCS = Var(domain=NonNegativeReals)

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
    eds_opex = 365 * sum(πc[c] * (cost[t] * model.PSp[t, c] * (Δt / 60)) for t in Ωt for c in Ωc)
    
    ev_opex = 365 * sum(πc[c] * (
        data["EVCS"]["discharging_cost"] * model.PEV_d[ev, t, c] * (Δt / 60))
        for ev in Ωev for t in Ωt for c in Ωc
    )
    tg_opex = 365 * sum(πc[c] * (
        data["TG"]["OPEX"] * model.PTG[t, c] * (Δt / 60))
        for t in Ωt for c in Ωc
    )
    pv_oem    = data['PV']['O&M'] * model.PPVmax * data["PV"]['CAPEX']
    bess_oem = 365 * sum(πc[c] * (
        data["BESS"]["charging_cost"] * model.PBESS_c[t, c] * (Δt / 60) +
        data["BESS"]["discharging_cost"] * model.PBESS_d[t, c] * (Δt / 60))
        for t in Ωt for c in Ωc
    )
    evcs_oem = data["EVCS"]["O&M"] * model.NEVCS * data["EVCS"]["CAPEX"]
    return model.OPEX == eds_opex + bess_oem + ev_opex + tg_opex  + pv_oem + evcs_oem
model.opex_constraint = Constraint(rule=opex_constraint_rule)


# Update power balance constraint
def power_balance_rule(model, t, c):
    pv = model.PPVmax * fs['5']['pv'][t]  # PV generation
    load = data["EDS"]["LOAD"] * fs['5']['load'][t]  # Demand
    return (
        model.PS[t, c] + pv + sum(model.PEV_d[ev, t, c] for ev in Ωev) + model.PBESS_d[t, c] + model.PTG[t, c] ==
        sum(model.PEV_c[ev, t, c] for ev in Ωev) + model.PBESS_c[t, c] + load
    )
model.power_balance = Constraint(Ωt, Ωc, rule=power_balance_rule)


################################################################################
############################# TG constraints ###################################
################################################################################

def tg_power_rule(model, t, c):
    return model.PTG[t, c] <= model.TG_MAX_CAP  # Power output cannot exceed maximum installed capacity
model.tg_power_constraint = Constraint(Ωt, Ωc, rule=tg_power_rule)


################################################################################
############################ EDS constraints ###################################
################################################################################
def eds_contingency_rule(model, t, c):
    if c == 0:
        return Constraint.Skip
    
    tt = datetime.strptime(t, "%H:%M")
    tc = datetime.strptime(c, "%H:%M")
    if tc <= tt < tc + timedelta(hours=data['off-grid']['duration']):
        return model.PS[t, c] == 0  # Substation power is zero during contingency
    else:
        return Constraint.Skip  # No constraint outside contingency periods
model.eds_contingency = Constraint(Ωt, Ωc, rule=eds_contingency_rule)

def eds_power_rule(model, t, c):
    return model.PS[t, c] == model.PSp[t, c] - model.PSn[t, c]
model.eds_power_constraint = Constraint(Ωt, Ωc, rule=eds_power_rule)


################################################################################
#########################   BESS constraints ###################################
################################################################################
def bess_energy_rule(model, t, c):
    return model.EBESS[t, c] <= model.EmaxBESS
model.bess_energy_constraint = Constraint(Ωt, Ωc, rule=bess_energy_rule)

def bess_power_rule_c(model, t, c):
    return model.PBESS_c[t, c] <= model.EmaxBESS * data["BESS"]['crate']
model.bess_power_c = Constraint(Ωt, Ωc, rule=bess_power_rule_c)

def bess_power_rule_d(model, t, c):
    return model.PBESS_d[t, c] <= model.EmaxBESS * data["BESS"]['crate']
model.bess_power_d = Constraint(Ωt, Ωc, rule=bess_power_rule_d)

def bess_energy_update_rule(model, t, c):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    η = 1  # BESS efficiency
    return model.EBESS[t2, c] == model.EBESS[t1, c] + η * model.PBESS_c[t2, c] * (Δt / 60) - model.PBESS_d[t2, c] * (η * Δt / 60)
model.bess_energy_update = Constraint(Ωt, Ωc, rule=bess_energy_update_rule)

def bess_charging_linearization_rule(model, t, c):
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    η = 1  # BESS efficiency
    return model.PBESS_c[t2, c] <= (model.EmaxBESS - model.EBESS[t1, c]) / (η * Δt / 60)
model.bess_charging_linearization = Constraint(Ωt, Ωc, rule=bess_charging_linearization_rule)

def bess_discharging_linearization_1_rule(model, t, c):
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    η = 1  # BESS efficiency
    return model.PBESS_d[t2, c] <= model.EBESS[t1, c] * η / (Δt / 60)
model.bess_discharging_linearization_1 = Constraint(Ωt, Ωc, rule=bess_discharging_linearization_1_rule)

def bess_discharging_linearization_2_rule(model, t, c):
    return model.PBESS_d[t, c] <= model.EmaxBESS * data["BESS"]['crate'] - model.PBESS_c[t, c]
model.bess_discharging_linearization_2 = Constraint(Ωt, Ωc, rule=bess_discharging_linearization_2_rule)

def bess_initial_energy_rule(model, c):
    return model.EBESS[Ωt[0], c] == data["BESS"]["SoCini"] * model.EmaxBESS
model.bess_initial_energy = Constraint(Ωc, rule=bess_initial_energy_rule)

def bess_offgrid_rule(model, t, c):
    if c != 0:
        tt = datetime.strptime(t, "%H:%M")
        tc = datetime.strptime(c, "%H:%M")
        if tt < tc:
            return model.EBESS[t, c] == model.EBESS[t, 0]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bess_offgrid = Constraint(Ωt, Ωc, rule=bess_offgrid_rule)



################################################################################
#########################   ALPHA constraints ###################################
################################################################################
def alpha_ev_update(model, ev, t, c):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")
    if ta < td and t0 > ta and t0 <= td:
        return model.αEV[ev, t, c] <= 1
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.αEV[ev, t, c] <= 1
    elif t0 == ta:
        return model.αEV[ev, t, c] <= 1
    else:
        return model.αEV[ev, t, c] <= 0
model.alpha_ev_update = Constraint(Ωev, Ωt, Ωc, rule=alpha_ev_update)


def ev_charging_power_rule(model, ev, t, c):
    return model.PEV_c[ev, t, c] <= model.αEV[ev, t, c] * EV[ev]['Pmax_c']
model.ev_charging_power_alpha = Constraint(Ωev, Ωt, Ωc, rule=ev_charging_power_rule)

def ev_discharging_power_rule(model, ev, t, c):
    return model.PEV_d[ev, t, c] <= model.αEV[ev, t, c] * EV[ev]['Pmax_d']
model.ev_discharging_power_alpha = Constraint(Ωev, Ωt, Ωc, rule=ev_discharging_power_rule)

def ev_connection_constraint_rule(model, ev, t, c):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    if ta < td and t0 > ta and t0 <= td:
        return model.αEV[ev, t2, c] >= model.αEV[ev, t1, c]
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.αEV[ev, t2, c] >= model.αEV[ev, t1, c]
    else:
        return Constraint.Skip
model.ev_connection_constraint = Constraint(Ωev, Ωt, Ωc, rule=ev_connection_constraint_rule)


# EVCS constraints
def evcs_constraint_rule(model, t, c):
    return model.NEVCS >= sum(model.αEV[ev, t, c] for ev in Ωev)
model.evcs_constraint = Constraint(Ωt, Ωc, rule=evcs_constraint_rule)

################################################################################
###########################   EV constraints ###################################
################################################################################

def ev_energy_update_rule(model, ev, t, c):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    η = EV[ev]['eff']
    if ta < td and t0 > ta and t0 <= td:
        return model.EEV[ev, t2, c] == model.EEV[ev, t1, c] + η * model.PEV_c[ev, t2, c] * (Δt / 60) - model.PEV_d[ev, t2, c] * (η * Δt / 60)
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.EEV[ev, t2, c] == model.EEV[ev, t1, c] + η * model.PEV_c[ev, t2, c] * (Δt / 60) - model.PEV_d[ev, t2, c] * (η * Δt / 60)
    elif t0 == ta:
        return model.EEV[ev, t, c] == EV[ev]['SoCini'] * EV[ev]['Emax']
    else:
        return model.EEV[ev, t, c] == 0
model.ev_energy_update = Constraint(Ωev, Ωt, Ωc, rule=ev_energy_update_rule)


def ev_energy_offgrid_rule(model, ev, t, c):
    if c != 0:
        tt = datetime.strptime(t, "%H:%M")
        tc = datetime.strptime(c, "%H:%M")
        if tt < tc:
            return model.EEV[ev, t, c] == model.EEV[ev, t, 0]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.ev_energy_offgrid = Constraint(Ωev, Ωt, Ωc, rule=ev_energy_offgrid_rule)

# Modified version of the V2G constraints that handles time correctly
def v2g_constraints(model, ev, t, c):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")   
    η = EV[ev]['eff']
    
    return model.PEV_c[ev, t2, c] <= (EV[ev]['Emax'] - model.EEV[ev, t1, c])  / (η * Δt / 60) 

model.v2g_charge = Constraint(Ωev, Ωt, Ωc, rule=v2g_constraints)

def v2g_discharge_constraints(model, ev, t, c):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    η = EV[ev]['eff']

    return model.PEV_d[ev, t2, c] <= (model.EEV[ev, t1, c] * η) / (Δt / 60)

model.v2g_discharge = Constraint(Ωev, Ωt, Ωc, rule=v2g_discharge_constraints)

def v2g_discharge_max(model, ev, t, c):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")

    # Discharge power should not exceed the calculated max value
    return model.PEV_d[ev, t2, c] <= EV[ev]['Pmax_d'] - (EV[ev]['Pmax_d'] / EV[ev]['Pmax_c']) * model.PEV_c[ev, t2, c]
model.v2g_discharge_max = Constraint(Ωev, Ωt, Ωc, rule=v2g_discharge_max)


def EV_departure_rule(model, ev, c):
    return model.EEV[ev, EV[ev]['departure'], c] == EV[ev]['Emax']
model.EV_departure = Constraint(Ωev, Ωc, rule=EV_departure_rule)



results = SolverFactory('gurobi').solve(model)

contingency_results = {}
for c in Ωc:
    contingency_results[c] = {
        'PS': [model.PS[t, c].value for t in Ωt],
        'PBESS_c': [model.PBESS_c[t, c].value for t in Ωt],
        'PBESS_d': [model.PBESS_d[t, c].value for t in Ωt],
        'PEV_c': [sum(model.PEV_c[ev, t, c].value for ev in Ωev) for t in Ωt],
        'PEV_d': [sum(model.PEV_d[ev, t, c].value for ev in Ωev) for t in Ωt],
        'PV': [model.PPVmax.value * fs['5']['pv'][t] for t in Ωt],
        'Demand': [data["EDS"]["LOAD"] * fs['5']['load'][t] for t in Ωt],
        'PTG': [model.PTG[t, c].value for t in Ωt]  # Gerador térmico
    }

# ==========================================
# Verificar status do solver e imprimir resultados
# ==========================================
print("Resultados de Otimização:")
if results.solver.status == SolverStatus.ok:
    print(f"Total Energy Cost: {value(model.objective):.2f} units")
    print(f"Maximum Solar Generation Capacity (PPVmax): {model.PPVmax.value:.2f} kW")
    print(f"Maximum BESS Energy Capacity (EmaxBESS): {model.EmaxBESS.value:.2f} kWh")
    print(f"Maximum Thermal Generator Capacity (TG_MAX_CAP): {model.TG_MAX_CAP.value:.2f} kW")
    print(f"Number of EV Charging Stations (NEVCS): {model.NEVCS.value:.0f}")
else:
    print("Solution not found")

# Garante que a pasta 'Results' exista (para evitar erro ao salvar)
os.makedirs("Results", exist_ok=True)

# ==========================================
# 1) PLOTS DE CADA CENÁRIO DE CONTINGÊNCIA
# ==========================================

colors = {
    'Substation': 'blue',
    'Demand': 'black',
    'Solar': 'orange',
    'Total EV charging Power': 'purple',
    'Total EV discharging Power': 'pink',
    # 'Net BESS Power': 'green',
    'Bess charging Power': 'green',
    'Bess discharging Power': 'yellow',
    'Thermal Generator': 'red'
}

for c in Ωc:
    # Garante que c seja string segura para nome de arquivo
    safe_c = str(c)
    safe_c = re.sub(r'[^a-zA-Z0-9_]+', '_', safe_c)

    PS_values = contingency_results[c]['PS']
    PBESS_c_values = contingency_results[c]['PBESS_c']
    PBESS_d_values = contingency_results[c]['PBESS_d']
    PEV_c_values = contingency_results[c]['PEV_c']
    PEV_d_values = contingency_results[c]['PEV_d']
    PV_generated = contingency_results[c]['PV']
    demand_values = contingency_results[c]['Demand']
    PTG_values = contingency_results[c]['PTG']
    
    # Calcula Net EV e Net BESS
    # net_EV_power = [PEV_c_values[t_idx] - PEV_d_values[t_idx] for t_idx in range(len(Ωt))]
    net_BESS_power = [PBESS_c_values[t_idx] - PBESS_d_values[t_idx] for t_idx in range(len(Ωt))]
    discharging = [-1*PBESS_d_values[t_idx] for t_idx in range(len(Ωt))]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Plot das variáveis (linhas)
    ax.plot(Ωt, PS_values, label='Substation Power (PS)', color=colors['Substation'], linewidth=2)
    ax.plot(Ωt, demand_values, label='Demand', color=colors['Demand'], linewidth=2, linestyle='--')
    ax.plot(Ωt, PV_generated, label='Solar Generation', color=colors['Solar'], linewidth=2)
    ax.plot(Ωt, PEV_c_values, label='Total EV charging Power (PEVc)', color=colors['Total EV charging Power'], linewidth=2)
    ax.plot(Ωt, PEV_d_values, label='Total EV discharging Power (PEVd)', color=colors['Total EV discharging Power'], linewidth=2)
    ax.plot(Ωt, PTG_values, label='Thermal Generator Power (PTG)', color=colors['Thermal Generator'], linewidth=2)
    
    # Plot do Net BESS Power como barras
    # ax.bar(Ωt, net_BESS_power, color=colors['Net BESS Power'], alpha=0.5,
    #        label='Net BESS Power (BESSc - BESSd)')
    ax.bar(Ωt, PBESS_c_values, color=colors['Bess charging Power'], alpha=0.5,
              label='BESS Charging Power (PBESSc)')
    ax.bar(Ωt, discharging, color=colors['Bess discharging Power'], alpha=0.5,
                label='BESS Discharging Power (PBESSd)')

    ax.set_title(f"Power in the System Over Time (Contingency {c})", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (HH:MM)", fontsize=12)
    ax.set_ylabel("Power (kW)", fontsize=12)
    ax.grid(True, which='both', linestyle=':', color='lightgray')
    ax.legend(loc='upper left', fontsize=10, frameon=True)

    # ========================================
    # Ajuste dos Ticks do eixo X
    # ========================================
    num_ticks = 10  # Definir um número razoável de ticks para melhor visualização
    tick_positions = list(range(0, len(Ωt), max(1, len(Ωt) // num_ticks)))  # Espaçamento uniforme
    tick_labels = [Ωt[i] for i in tick_positions]  # Apenas os valores selecionados

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')  # Rotaciona para melhor leitura

    plt.tight_layout()
    plt.savefig(f"Results/Contingency_{safe_c}_Power_Graph.png", dpi=300)
    plt.close()


# ==========================================
# 2) HEATMAP DE SoC PARA EV E BESS (para cada c)
# ==========================================
# Função para converter string "HH:MM" para objeto time
def to_time(hhmm):
    return datetime.strptime(hhmm, "%H:%M").time()

# Função que verifica se o EV está PRESENTE naquele instante `t`
def is_ev_present(t: datetime.time, arrival: datetime.time, departure: datetime.time) -> bool:
    if arrival < departure:
        return arrival <= t <= departure
    else:
        return not (departure <= t <= arrival)  # O contrário do período de ausência



colorscale = [
    (0.0, 'lightyellow'),  # SoC=0
    (0.5, 'orange'),       # SoC=0.5
    (1.0, 'darkred')       # SoC=1
]
cmap_custom = LinearSegmentedColormap.from_list('CustomSoC', colorscale)

# ==========================================
# Loop para cada contingência c
# ==========================================
for c in Ωc:
    safe_c = re.sub(r'[^a-zA-Z0-9_]+', '_', str(c))

    # 1) Cria DataFrame de SoC (EV)
    ev_soc_values = {
        ev: [model.EEV[ev, t, c].value / EV[str(ev)]['Emax'] for t in Ωt]
        for ev in Ωev
    }
    df_ev_soc = pd.DataFrame(ev_soc_values, index=Ωt)

    # 2) Cria DataFrame de αEV (EV)
    alpha_ev_values = {
        ev: [model.αEV[ev, t, c].value for t in Ωt]
        for ev in Ωev
    }
    df_ev_alpha = pd.DataFrame(alpha_ev_values, index=Ωt)

    # 3) Cria DataFrame de SoC (BESS)
    if model.EmaxBESS.value > 0:
        bess_soc_values = {
            'BESS': [model.EBESS[t, c].value / model.EmaxBESS.value for t in Ωt]
        }
        df_bess_soc = pd.DataFrame(bess_soc_values, index=Ωt)
    else:
        df_bess_soc = pd.DataFrame(0, index=Ωt, columns=['BESS'])

    # 4) Concatena EV + BESS para plotar SoC
    df_combined = pd.concat([df_ev_soc, df_bess_soc], axis=1)

    # ==========================================
    # Construindo o DataFrame de anotações
    # ==========================================
    df_annotations = pd.DataFrame(index=df_combined.index, columns=df_combined.columns, dtype=object)

    # Convertendo Ωt para objetos time (para comparar com arrival/departure)
    times_as_time = [to_time(tt) for tt in Ωt]

    # Obtém arrival/departure em formato time
    arrival_time = {}
    departure_time = {}
    for ev in Ωev:
        arr_str = EV[str(ev)]['arrival']
        dep_str = EV[str(ev)]['departure']
        arrival_time[ev] = to_time(arr_str)
        departure_time[ev] = to_time(dep_str)

    # Preenche cada célula apenas quando o EV está presente
    for row_idx, t_str in enumerate(Ωt):
        t_time = times_as_time[row_idx]
        for col in df_combined.columns:
            soc_val = df_combined.loc[t_str, col]  # SoC
            
            # Se for BESS, apenas exibe o SoC
            if col == 'BESS':
                df_annotations.loc[t_str, col] = f"{soc_val*100:.1f}%"
            else:
                # 'col' é o ID do EV
                alpha_val = df_ev_alpha.loc[t_str, col]
                arr_t = arrival_time[col]
                dep_t = departure_time[col]

                # Apenas exibir se EV está presente
                if is_ev_present(t_time, arr_t, dep_t):
                    df_annotations.loc[t_str, col] = f"{soc_val*100:.1f}% | α={int(alpha_val)}"
                else:
                    df_annotations.loc[t_str, col] = ""

    # ==========================================
    # Plot do heatmap de SoC
    # ==========================================
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
        annot_kws={'size': 6, 'rotation': 90},  # Fonte menor e rotação
        ax=ax
    )

    ax.set_title(f"EV & BESS SoC c/ αEV - Contingency {c}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (HH:MM)", fontsize=12)
    ax.set_ylabel("EVs and BESS", fontsize=12)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"Results/SoC_and_alphaEV_Heatmap_{safe_c}.png", dpi=300)
    plt.close()

