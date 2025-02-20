import plotly.graph_objects as go
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, SolverFactory, Binary
import pyomo.environ as pyo
from datetime import datetime, timedelta
import matplotlib as mpl    
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

folder = "Marcelo_Data"


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


# Definindo o modelo
model = ConcreteModel()


# Variable definition
model.PS = Var(Ωt, domain=NonNegativeReals)  # Energia comprada da rede

# EV operation variables
model.SoCEV = Var(Ωev, Ωt, domain=NonNegativeReals)  # EV State of Charge
model.PEV_c = Var(Ωev, Ωt, domain=NonNegativeReals)  # EV charging power
# model.PEV_d = Var(Ωev, Ωt, domain=NonNegativeReals)  # EV discharging power
model.αEV   = Var(Ωev, Ωt, domain=Binary)  # EV charging factor

# PV variable
model.PPVmax = Var(domain=NonNegativeReals)          # Maximum PV generation


# Economic variables
model.CAPEX = Var(within=NonNegativeReals)  # Total CAPEX
model.OPEX = Var(within=NonNegativeReals)  # Total OPEX

#Objective function
def objective_rule(model):
    total_opex = 0
    for n in range(1, data["OPEX"]["years"] + 1):
        total_opex += model.OPEX * 365 *( (1 + data["OPEX"]["rate"]) ** (n - 1))
    return model.CAPEX + total_opex
model.objective = Objective(rule=objective_rule, sense=pyo.minimize)

# CAPEX constraint (ensuring CAPEX is the sum of InstPV and InstWT)
def capex_constraint_rule(model):
    PV = model.PPVmax * data["CAPEX"]['PV']  # PV installation cost
    return model.CAPEX == PV
model.capex_constraint = Constraint(rule=capex_constraint_rule)

# OPEX constraint (sum of prices * PS for each time period t)
def opex_constraint_rule(model):
    return model.OPEX == sum(cost[t] * model.PS[t] for t in Ωt)
model.opex_constraint = Constraint(rule=opex_constraint_rule)


def alpha_ev_update(model, ev, t):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M") 
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")
    
    if ta < td and t0 > ta and t0 <= td:
        return model.αEV[ev, t] <= 1
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.αEV[ev, t] <= 1
    elif t0 == ta:
        return model.αEV[ev, t] <= 1
    else:
        return model.αEV[ev, t] <= 0
model.alpha_ev_update = Constraint(Ωev ,Ωt, rule=alpha_ev_update)

def ev_power_rule(model, ev, t):
    return model.PEV_c[ev, t] <= model.αEV[ev, t] * EV[ev]['Emax']
model.ev_power = Constraint(Ωev, Ωt, rule=ev_power_rule)

def socev_update(model, ev, t):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M") 
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")

    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    
    if ta < td and t0 > ta and t0 <= td:
        return model.SoCEV[ev, t2] == model.SoCEV[ev, t1] + model.PEV_c[ev, t2] * (Δt/60)
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.SoCEV[ev, t2] == model.SoCEV[ev, t1] + model.PEV_c[ev, t2] * (Δt/60)
    elif t0 == ta:
        return model.SoCEV[ev, t] == EV[ev]['SoCini'] * EV[ev]['Emax']
    else:
        return model.SoCEV[ev, t] == 0
model.socev_update = Constraint(Ωev ,Ωt, rule=socev_update)

def EV_departure_rule(model, ev):
    return model.SoCEV[ev, EV[ev]['departure']] == EV[ev]['Emax']
model.EV_departure = Constraint(Ωev, rule=EV_departure_rule)



def limite_energia(model, t):
    return model.PS[t] <= data["EDS"]["Pmax"]  # Exemplo: Capacidade máxima de 10 kWh por hora
model.limite_energia = Constraint(Ωt, rule=limite_energia)


# Balanço de carga: PS = PEV + Demanda - PV*Irradiância - PWT*Vento
def balanco_carga(model, t):
    pv = model.PPVmax * fs['5']['pv'][t]  # Geração fotovoltaica
    load = data["EDS"]["LOAD"] * fs['5']['load'][t]  # Demanda
    return model.PS[t] + pv == sum(model.PEV_c[ev, t] for ev in Ωev) + load
model.balanco_carga = Constraint(Ωt, rule=balanco_carga)



# Resolvendo o modelo
results = SolverFactory('gurobi').solve(model)

# Exibindo os resultados
print("Resultados de Otimização:")

# Check solver status
if results.solver.status == pyo.SolverStatus.ok:
    # Display results
    print(f"Total Energy Cost: {pyo.value(model.objective):.2f} units")
    print(f"Maximum Solar Generation Capacity (PPVmax): {model.PPVmax.value:.2f} kW")
    
    # Number of time periods (using Ωt to determine the range)
    T = len(Ωt)  # Define T as the length of Ωt, which is the number of time steps
    e = len(Ωev)
    #for ev in range(e):
        #for t in range(T):
            #print(f"Hour {t}: EV {ev} SoC = {model.SoCEV[ev, t].value:.2f} kWh, Energy Bought = {model.PS[t].value:.2f} kWh, Energy Supplied = {model.PEV_c[ev, t].value:.2f} kWh")
else:
    print("Solution not found")



# Exibindo os resultados de CAPEX e OPEX
#print(f"CAPEX (Custo de Instalação Total): {CAPEX:.2f} unidades")
#print(f"OPEX (Custo Operaçãp): {OPEX:.2f} unidades")
#print(f"OPEX (Custo Operacional Anual): {OPEX_anual:.2f} unidades")

# Resultados do modelo de otimização (valores de SoC, PS, PEV, InstPV e InstWT de cada hora)

# Creating graphs with Plotly
fig = go.Figure()

# SoC graph (State of Charge)
SoC_values = [model.SoCEV[ev, t].value for ev in Ωev for t in Ωt]  # Extract SoC for each EV and each time
fig.add_trace(go.Scatter(x=[f"{ev}-{t}" for ev in Ωev for t in Ωt], y=SoC_values, mode='lines+markers', name='SoC (kWh)', line=dict(color='blue')))

# PS graph (Energy Bought)
PS_values = [model.PS[t].value for t in Ωt]  # Extract energy bought for each time step
fig.add_trace(go.Scatter(x=Ωt, y=PS_values, mode='lines+markers', name='Energy Bought (PS) (kWh)', line=dict(color='red')))

# PEV graph (Energy Supplied by EV)
PEV_values = [model.PEV_c[ev, t].value for ev in Ωev for t in Ωt]  # Extract energy supplied by EV for each EV and each time
fig.add_trace(go.Scatter(x=[f"{ev}-{t}" for ev in Ωev for t in Ωt], y=PEV_values, mode='lines+markers', name='Energy Supplied (PEV) (kWh)', line=dict(color='green')))

# Solar Generation graph
PV_generated = [model.PPVmax.value * fs['5']['pv'][t] for t in Ωt]  # Extract solar generation for each time step
fig.add_trace(go.Scatter(x=Ωt, y=PV_generated, mode='lines+markers', name='Solar Generation (kWh)', line=dict(color='orange')))

# Add trace for demand
demand_values = [data["EDS"]["LOAD"] * fs['5']['load'][t] for t in Ωt]  # Extract demand for each time step
fig.add_trace(go.Scatter(x=Ωt, y=demand_values, mode='lines+markers', name='Demand (kWh)', line=dict(color='black', dash='dot')))


# Layout configuration
fig.update_layout(
    title="State of Charge (SoC), Energy Bought (PS), Energy Supplied (PEV), Solar and Wind Generation Over Hours",
    xaxis_title="Time of Day",
    yaxis_title="Value (kWh)",
    legend_title="Variables",
    template="plotly",
    showlegend=True
)

# Displaying the graph
fig.show()


