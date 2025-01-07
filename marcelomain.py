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

if results.solver.status == pyo.SolverStatus.ok:
    print(f"Custo Total de Energia: {pyo.value(model.objective):.2f} unidades")
    print(f"Capacidade Máxima de Geração Solar (PPVmax): {model.PPVmax.value:.2f} kW")
    print(f"Capacidade Máxima de Geração Eólica (PWTmax): {model.PWTmax.value:.2f} kW")
    for t in range(T):
        print(f"Hora {t}: SoC = {model.SoCEV[t].value:.2f} kWh, Energia Comprada = {model.PS[t].value:.2f} kWh, Energia Fornecida = {model.PEV[t].value:.2f} kWh")
else:
    print("Solução não encontrada")


# Exibindo os resultados de CAPEX e OPEX
#print(f"CAPEX (Custo de Instalação Total): {CAPEX:.2f} unidades")
#print(f"OPEX (Custo Operaçãp): {OPEX:.2f} unidades")
#print(f"OPEX (Custo Operacional Anual): {OPEX_anual:.2f} unidades")

# Resultados do modelo de otimização (valores de SoC, PS, PEV, InstPV e InstWT de cada hora)
horas = list(range(T))  # Horas de 0 a 23

# Extraindo os valores de SoC, PS, PEV, InstPV, InstWT do modelo
SoC_values = [model.SoCEV[t].value for t in range(T)]
PS_values = [model.PS[t].value for t in range(T)]
PEV_values = [model.PEV[t].value for t in range(T)]


# Calculando a geração solar e eólica para cada hora
PV_gerado = [model.PPVmax.value * irradiancia_solar[t] for t in range(T)]  # Geração fotovoltaica
WT_gerado = [model.PWTmax.value * vento[t] for t in range(T)]  # Geração eólica

# Criando gráficos com Plotly
fig = go.Figure()

# Gráfico de SoC
fig.add_trace(go.Scatter(x=horas, y=SoC_values, mode='lines+markers', name='SoC (kWh)', line=dict(color='blue')))

# Gráfico de PS (Energia Comprada)
fig.add_trace(go.Scatter(x=horas, y=PS_values, mode='lines+markers', name='Energia Comprada (PS) (kWh)', line=dict(color='red')))

# Gráfico de PEV (Energia Fornecida pelo EV)
fig.add_trace(go.Scatter(x=horas, y=PEV_values, mode='lines+markers', name='Energia Fornecida (PEV) (kWh)', line=dict(color='green')))

# Gráfico de Geração Solar
fig.add_trace(go.Scatter(x=horas, y=PV_gerado, mode='lines+markers', name='Geração Solar (kWh)', line=dict(color='orange')))

# Gráfico de Geração Eólica
fig.add_trace(go.Scatter(x=horas, y=WT_gerado, mode='lines+markers', name='Geração Eólica (kWh)', line=dict(color='purple')))

# Add trace for demanda
fig.add_trace(go.Scatter(x=horas, y=[demanda[t] for t in horas], mode='lines+markers', name='Demanda (kWh)', line=dict(color='black', dash='dot')))


# Configuração do layout
fig.update_layout(
    title="Estado de Carga (SoC), Energia Comprada (PS), Energia Fornecida (PEV), Geração Solar e Geração Eólica ao Longo das Horas",
    xaxis_title="Hora do Dia",
    yaxis_title="Valor (kWh)",
    legend_title="Variáveis",
    template="plotly",
    showlegend=True
)

# Exibindo o gráfico
fig.show()
# Verificando os preços para cada hora

