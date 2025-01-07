import plotly.graph_objects as go
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, SolverFactory
import numpy as np
import pyomo.environ as pyo

from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import matplotlib as mpl    
import seaborn as sns
import pandas as pd

import json
import os


# Definindo o modelo
model = ConcreteModel()

# Número de intervalos de tempo (24 horas)
T = 24

# Parâmetros de entrada
precos = np.random.uniform(10, 15, size=T)  # Preço aleatório por kWh
demanda = np.random.uniform(20, 40, size=T)  # Demanda aleatória por hora (em kWh)
geracao_solar = np.random.uniform(0, 1, size=T)  # Geração solar aleatória entre 0 e 1 kWh por hora
irradiancia_solar = np.random.uniform(0, 1, size=T)  # Irradiância solar (0 a 1) em cada hora
vento = np.random.uniform(0, 1, size=T)  # Velocidade do vento (0 a 1) em cada hora



# Definindo as variáveis no modelo
model.SoCEV = Var(range(T), domain=NonNegativeReals, initialize=0)  # Estado de Carga do EV
model.PS = Var(range(T), domain=NonNegativeReals, initialize=0)    # Energia comprada da rede
model.PEV = Var(range(T), domain=NonNegativeReals, initialize=0)   # Energia fornecida pelo EV
model.PPVmax = Var(domain=NonNegativeReals, initialize=10)        # Capacidade máxima de geração solar (exemplo 10 kW)
model.PWTmax = Var(domain=NonNegativeReals, initialize=10)        # Capacidade máxima de geração eólica (exemplo 10 kW)


# Constantes
cPV = 50  # Custo fixo por unidade de capacidade instalada (em $/kW, por exemplo)
cWT = 60  # Custo fixo por unidade de capacidade instalada para eólica

# Variables
model.CAPEX = Var(within=NonNegativeReals)  # Total CAPEX
model.OPEX = Var(within=NonNegativeReals)  # Total OPEX


# Calculando CAPEX e OPEX

# CAPEX constraint (ensuring CAPEX is the sum of InstPV and InstWT)
def capex_constraint(model):
    custo_PV = model.PPVmax*cPV
    custo_WT = model.PWTmax*cWT
    return model.CAPEX == custo_PV + custo_WT

model.capex_constraint = Constraint(rule=capex_constraint)

# OPEX constraint (sum of prices * PS for each time period t)
def opex_constraint(model):
    return model.OPEX == sum(precos[t] * model.PS[t] for t in range(T))

model.opex_constraint = Constraint(rule=opex_constraint)


# Taxa de inflação (r_rate)
r_rate = 0.05  # 5% de inflação anual
n_years = 30  # Número de anos
# O cálculo de OPEX ajustado por inflação para cada ano
def custo_opex_com_inflacao(model): 
    custo_total_com_inflacao = 0

    # Somando os custos de OPEX ajustados pela inflação para cada ano
    for n in range(1, n_years + 1):
        OPEX_ano = model.OPEX*365 *( (1 + r_rate) ** (n - 1))  # Ajustando pela inflação para o ano n
        custo_total_com_inflacao += OPEX_ano  # Adicionando o OPEX do ano n ao total
    
    return custo_total_com_inflacao

# Definindo a função objetivo: Minimizar o custo de compra de energia da rede + custos de instalação
def custo_total(model):
    OPEX_inflacionado = custo_opex_com_inflacao(model)  # Calculando OPEX com inflação
    return model.CAPEX + OPEX_inflacionado

model.objective = Objective(rule=custo_total, sense=pyo.minimize)
# Restrições

# Dados dos veículos (com SoCini e Emax)
dados_veiculos = {
    "1": {
        "arrival": 20,  # Hora de chegada (22h)
        "departure": 6,  # Hora de saída (03h)
        "SoCini": 0.3,  # Estado de carga inicial (30%)
        "Emax": 40,     # Capacidade máxima do EV em kWh
    }
}

# Load the JSON file
with open('data/cost.json') as file:
    cost = json.load(file)

with open('data/scenarios.json') as file:
    fs = json.load(file)

with open('data/parameters.json') as file:
    data = json.load(file)
    
with open('data/EVs.json') as file:
    EV = json.load(file)

with open('data/EVCSs.json') as file:
    EVCS = json.load(file)

def socev_update(model, t):
    if t > 6 and t < 20:
        return model.SoCEV[t] == 0
    if t == dados_veiculos["1"]["arrival"]:
        return model.SoCEV[t] == dados_veiculos["1"]["SoCini"] * dados_veiculos["1"]["Emax"]
    elif t != 0:
        return model.SoCEV[t] == model.SoCEV[t-1] + model.PEV[t-1]  # SoCEV(t) = SoCEV(t-1) + PEV(t-1)
    
model.socev_update = Constraint(range(1, T), rule=socev_update)

def PEV_zero(model, t):
    if t > 6 and t < 20:
        return model.PEV[t] == 0
    else:
        Constraint.Skip()

    
model.PEV_zero = Constraint(range(7, 19), rule=PEV_zero)

def carga_completa(model):
    return model.SoCEV[dados_veiculos["1"]["departure"]] == dados_veiculos["1"]["Emax"]

model.carga_completa = Constraint(rule=carga_completa)

def limite_energia(model, t):
    return model.PS[t] <= 50  # Exemplo: Capacidade máxima de 10 kWh por hora

model.limite_energia = Constraint(range(T), rule=limite_energia)

def socev_midnight(model):
    return model.SoCEV[0] == model.SoCEV[(T-1)] + model.PEV[(T-1)]

model.socev_midnight = Constraint(rule=socev_midnight)

# Balanço de carga: PS = PEV + Demanda - PV*Irradiância - PWT*Vento
def balanco_carga(model, t):
    PV_gerado = model.PPVmax * irradiancia_solar[t]  # Geração fotovoltaica
    WT_gerado = model.PWTmax * vento[t]  # Geração eólica   
    return model.PS[t] + PV_gerado + WT_gerado == model.PEV[t] + demanda[t] 

model.balanco_carga = Constraint(range(T), rule=balanco_carga)



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

