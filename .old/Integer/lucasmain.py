import plotly.graph_objects as go
from pyomo.environ import *
import pyomo.environ as pyo
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for saving plots without needing Tkinte   
import matplotlib.pyplot as plt  # Importing pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

folder = "Sizing/Marcelo_Data"


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
model.PS  = Var(Ωt, domain=Reals)  # Energia comprada da rede
model.PSp = Var(Ωt, domain=NonNegativeReals)  # Energia comprada da rede
model.PSn = Var(Ωt, domain=NonNegativeReals, bounds=(0, -data["EDS"]['Pmin']))  # Energia comprada da rede

# EV operation variables
model.SoCEV = Var(Ωev, Ωt, domain=NonNegativeReals)  # EV State of Charge
model.PEV_c = Var(Ωev, Ωt, domain=NonNegativeReals)  # EV charging power
model.PEV_d = Var(Ωev, Ωt, domain=NonNegativeReals)  # EV discharging power
model.αEV   = Var(Ωev, Ωt, domain=Binary)  # EV charging factor
model.γEV_c = Var(Ωev, Ωt, domain=Binary)  # EV charging status
model.γEV_d = Var(Ωev, Ωt, domain=Binary)  # EV discharging status

# Number of charging stations
model.NEVCS = Var(domain=NonNegativeIntegers)# Number of charging stations


# BESS variables
model.PBESSmax = Var(domain=NonNegativeReals)  # BESS power
model.EBESSmax = Var(domain=NonNegativeReals)  # BESS energy
model.SoCBESS = Var(Ωt, domain=NonNegativeReals)   # BESS State of Charge
model.PBESS_c = Var(Ωt, domain=NonNegativeReals)   # BESS charging power
model.PBESS_d = Var(Ωt, domain=NonNegativeReals)   # BESS discharging power
model.γBESS_c = Var(Ωt, domain=Binary)             # BESS charging status
model.γBESS_d = Var(Ωt, domain=Binary)             # BESS discharging status

# Thermal Generator variables
model.PTGmax = Var(domain=NonNegativeReals)    # Thermal Generator power
model.PTG = Var(Ωt, domain=NonNegativeReals)   # Thermal Generator power

# PV variable
model.PPVmax = Var(domain=NonNegativeReals)        # Maximum PV generation


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
    PV    = model.PPVmax   * data["CAPEX"]['PV']  # PV installation cost
    BESS  = model.EBESSmax * data["CAPEX"]['BESS']  # BESS installation cost
    TG    = model.PTGmax   * data["CAPEX"]['TG']  # Thermal Generator installation cost
    NEVCS = model.NEVCS    * data["CAPEX"]['EVCS'] # Total CAPEX
    return model.CAPEX == PV + TG + BESS + NEVCS
model.capex_constraint = Constraint(rule=capex_constraint_rule)

# OPEX constraint (sum of prices * PS for each time period t)
def opex_constraint_rule(model):
    return model.OPEX == sum(cost[t] * model.PSp[t] + data['TG']['cost'] * model.PTG[t] for t in Ωt)
model.opex_constraint = Constraint(rule=opex_constraint_rule)

# Balanço de carga: PS = PEV + Demanda - PV*Irradiância - PWT*Vento
def power_balance_rule(model, t):
    pv = model.PPVmax * fs['5']['pv'][t]  # Geração fotovoltaica
    load = data["EDS"]["LOAD"] * fs['5']['load'][t]  # Demanda
    return model.PS[t] + pv + sum(model.PEV_d[ev, t] for ev in Ωev) + model.PBESS_d[t] + model.PTG[t] == \
        sum(model.PEV_c[ev, t] for ev in Ωev) + load + model.PBESS_c[t]
model.power_balance = Constraint(Ωt, rule=power_balance_rule)

def eds_positive_power_limit_rule(model, t):
    return model.PS[t] <= data["EDS"]["Pmax"]  # Exeplto: Capacidade máxima de 10 kWh por hora
model.eds_positive_power_limit = Constraint(Ωt, rule=eds_positive_power_limit_rule)

def eds_negative_power_limit_rule(model, t):
    return model.PS[t] >= data["EDS"]["Pmin"]  # Exemplo: Capacidade máxima de 10 kWh por hora
model.eds_negative_power_limit = Constraint(Ωt, rule=eds_negative_power_limit_rule)

def eds_power_rule(model, t):
    return model.PS[t] == model.PSp[t] - model.PSn[t]
model.eds_power = Constraint(Ωt, rule=eds_power_rule)

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

def ev_charging_power_rule(model, ev, t):
    return model.PEV_c[ev, t] <= model.αEV[ev, t] * EV[ev]['Pmax_c']
model.ev_charging_power_alpha = Constraint(Ωev, Ωt, rule=ev_charging_power_rule)

def ev_discharging_power_rule(model, ev, t):
    return model.PEV_d[ev, t] <= model.αEV[ev, t] * EV[ev]['Pmax_d']
model.ev_discharging_power_alpha = Constraint(Ωev, Ωt, rule=ev_discharging_power_rule)

def ev_connection_constraint_rule(model, ev, t):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    if ta < td and t0 > ta and t0 <= td:
        return model.αEV[ev, t2] >= model.αEV[ev, t1]
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.αEV[ev, t2] >= model.αEV[ev, t1]
    else:
        return Constraint.Skip
model.ev_connection_constraint = Constraint(Ωev, Ωt, rule=ev_connection_constraint_rule)

def socev_update(model, ev, t):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")

    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")

    η = EV[ev]['eff']
    if ta < td and t0 > ta and t0 <= td:
        return model.SoCEV[ev, t2] == model.SoCEV[ev, t1] + η * model.PEV_c[ev, t2] * (Δt / 60) - model.PEV_d[ev, t2] * (η * Δt / 60)
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.SoCEV[ev, t2] == model.SoCEV[ev, t1] + η * model.PEV_c[ev, t2] * (Δt / 60) - model.PEV_d[ev, t2] * (η * Δt / 60)
    elif t0 == ta:
        return model.SoCEV[ev, t] == EV[ev]['SoCini'] * EV[ev]['Emax']
    else:
        return model.SoCEV[ev, t] == 0
model.socev_update = Constraint(Ωev, Ωt, rule=socev_update)

def EV_departure_rule(model, ev):
    return model.SoCEV[ev, EV[ev]['departure']] == EV[ev]['Emax']
model.EV_departure = Constraint(Ωev, rule=EV_departure_rule)

# Modified version of the V2G constraints that handles time correctly
def v2g_charge_power_limit_constraints_rule(model, ev, t):
    return model.PEV_c[ev, t] <= EV[ev]['Pmax_c'] * model.γEV_c[ev, t]
model.v2g_charge_power_limit = Constraint(Ωev, Ωt, rule=v2g_charge_power_limit_constraints_rule)

def v2g_discharge_power_limit_constraints_rule(model, ev, t): 
    return model.PEV_d[ev, t] <= EV[ev]['Pmax_d'] * model.γEV_d[ev, t]
model.v2g_discharge_power_limit = Constraint(Ωev, Ωt, rule=v2g_discharge_power_limit_constraints_rule)

def v2g_binnary_rule(model, ev, t):
    return model.γEV_c[ev, t] + model.γEV_d[ev, t] <= 1
model.v2g_binnary = Constraint(Ωev, Ωt, rule=v2g_binnary_rule)


# EVCS constraints
def evcs_constraint_rule(model, t):
    return model.NEVCS >= sum(model.αEV[ev, t] for ev in Ωev)
model.evcs_constraint = Constraint(Ωt, rule=evcs_constraint_rule)


# TG constraints
def tg_power_rule(model, t):
    return model.PTG[t] <= model.PTGmax
model.tg_power = Constraint(Ωt, rule=tg_power_rule)

#BESS constraints
def bess_charging_power_rule(model, t):
    return model.PBESS_c[t] <= model.PBESSmax
model.bess_charging_power = Constraint(Ωt, rule=bess_charging_power_rule)

def bess_discharging_power_rule(model, t):
    return model.PBESS_d[t] <= model.PBESSmax
model.bess_discharging_power = Constraint(Ωt, rule=bess_discharging_power_rule)

def bess_energy_rule(model, t):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    η = data["BESS"]["eff"]
    return model.SoCBESS[t2] == model.SoCBESS[t1] + η * model.PBESS_c[t2] * (Δt / 60) - model.PBESS_d[t2] * (Δt / 60 * η)
model.bess_energy = Constraint(Ωt, rule=bess_energy_rule)

def bess_initial_energy_rule(model):
    return model.SoCBESS[Ωt[0]] == data["BESS"]["SoCini"] * model.EBESSmax
model.bess_initial_energy = Constraint(rule=bess_initial_energy_rule)

def bess_final_energy_rule(model):
    return model.SoCBESS[Ωt[-1]] == data["BESS"]["SoCini"] * model.EBESSmax
model.bess_final_energy = Constraint(rule=bess_final_energy_rule)

def bess_charging_power_binary_rule(model, t):
    M = 1e6
    return model.PBESS_c[t] <= model.γBESS_c[t] * M
model.bess_charging_power_binary = Constraint(Ωt, rule=bess_charging_power_binary_rule)

def bess_discharging_power_binary_rule(model, t):
    M = 1e6
    return model.PBESS_d[t] <= model.γBESS_d[t] * M
model.bess_discharging_power_binary = Constraint(Ωt, rule=bess_discharging_power_binary_rule)

def bess_charging_status_rule(model, t):
    return model.γBESS_c[t] + model.γBESS_d[t] <= 1
model.bess_charging_status = Constraint(Ωt, rule=bess_charging_status_rule)

def bess_crate_rule(model):
    return model.PBESSmax == data["BESS"]["crate"] * model.EBESSmax
model.bess_crate = Constraint(rule=bess_crate_rule)

# Resolvendo o modelo
results = SolverFactory('gurobi').solve(model)

# Exibindo os resultados
print("Resultados de Otimização:")

# Check solver status
if results.solver.status == pyo.SolverStatus.ok:
    # Display results
    print(f"Total Cost: {pyo.value(model.objective)/1e6:.2f} MUSD")
    print(f"Maximum Solar Generation Capacity (PPVmax): {model.PPVmax.value:.2f} kW")
    print(f"Maximum BESS Power (PBESSmax): {model.PBESSmax.value:.2f} kW")
    print(f"Maximum BESS Energy (EBESSmax): {model.EBESSmax.value:.2f} kWh")
    print(f"Maximum Thermal Generator Power (PTGmax): {model.PTGmax.value:.2f} kW")
    
    print(f"Number of EVCS: {model.NEVCS.value:.0f}")
    # Number of time periods (using Ωt to determine the range)
    T = len(Ωt)  # Define T as the length of Ωt, which is the number of time steps
    e = len(Ωev)
    #for ev in range(e):
        #for t in range(T):
            #print(f"Hour {t}: EV {ev} SoC = {model.SoCEV[ev, t].value:.2f} kWh, Energy Bought = {model.PS[t].value:.2f} kWh, Energy Supplied = {model.PEV_c[ev, t].value:.2f} kWh")
else:
    print("Solution not found")





# Creating graphs with Plotly
fig = go.Figure()


# PS graph (Energy Bought)
PS_values = [model.PS[t].value for t in Ωt]  # Extract energy bought for each time step
fig.add_trace(go.Scatter(x=Ωt, y=PS_values, mode='lines+markers', name='Energy Bought (PS) (kWh)', line=dict(color='red')))


# Solar Generation graph
PV_generated = [model.PPVmax.value * fs['5']['pv'][t] for t in Ωt]  # Extract solar generation for each time step
fig.add_trace(go.Scatter(x=Ωt, y=PV_generated, mode='lines+markers', name='Solar Generation (kWh)', line=dict(color='orange')))

# Add trace for demand
demand_values = [data["EDS"]["LOAD"] * fs['5']['load'][t] for t in Ωt]  # Extract demand for each time step
fig.add_trace(go.Scatter(x=Ωt, y=demand_values, mode='lines+markers', name='Demand (kWh)', line=dict(color='black', dash='dot')))


# Ensure the Results folder exists
if not os.path.exists('Results'):
    os.makedirs('Results')

if not os.path.exists('Results/scenarios'):
    os.makedirs('Results/scenarios')

# Import matplotlib with the alias 'plt'
plt.rc('font', family='serif', serif='cmr10')
plt.rcParams['axes.unicode_minus'] = False

# Create a figure with subplots to plot all the heatmaps
fig, axes = plt.subplots(len(Ωev), 1, figsize=(16, 4 * len(Ωev)))

# Loop through each Electric Vehicle (EV) for SoCEV heatmap plotting
for ev, ax in zip(Ωev, axes):
    dic_values = {}  # To store numerical values (SoC of EV)
    dic_annotations = {}  # To store annotations with EV id and SoC percentage

    # We will now loop through time steps (Ωt) for each EV
    values = []  # List to store SoC values for each time step
    annotations = []  # List to store annotations

    for t in Ωt:  # Loop over time steps
        # Get the SoC of the EV at time t
        soc = pyo.value(model.SoCEV[ev, t])/EV[ev]['Emax']  # Correctly referencing SoCEV[ev, t] now
        annotation = f'{ev}\n{int(soc * 100)}%'  # Annotation with EV id and SoC percentage
        values.append(soc)  # Append the SoC value
        annotations.append(annotation)  # Append the annotation

    dic_values[f'EV {ev}'] = values  # Store SoC values for the EV
    dic_annotations[f'EV {ev}'] = annotations  # Store annotations for the EV

    # Create a DataFrame for SoC values
    df_values = pd.DataFrame.from_dict(dic_values, orient='index', columns=pd.to_datetime(Ωt, format='%H:%M').time)
    
    # Create a DataFrame for annotations
    df_annotations = pd.DataFrame.from_dict(dic_annotations, orient='index', columns=pd.to_datetime(Ωt, format='%H:%M').time)

    # Plotting the heatmap
    sns.heatmap(df_values, cmap='coolwarm', cbar_kws={'label': 'State of Charge (SoC)'},
                linewidths=.5, annot=df_annotations, fmt='', annot_kws={'size': 8}, ax=ax)

    # Annotating with EV id and SoC percentage
    for y, row in enumerate(df_annotations.values):
        for x, cell in enumerate(row):
            ev_id, soc_percent = cell.split('\n')
            ax.text(x + 0.5, y + 0.3, ev_id, ha='center', va='center', fontsize=8, color='black')  # EV id
            ax.text(x + 0.5, y + 0.7, soc_percent, ha='center', va='center', fontsize=6, color='gray')  # SoC percentage

    ax.set_title(f'State of Charge (SoCEV) - EV {ev}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Electric Vehicle (EV)')
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability
    ax.tick_params(axis='y', rotation=0)  # Keep y-axis labels horizontal

# Adjust layout for tight fitting
plt.tight_layout()

# Save all heatmaps into a single PNG
plt.savefig(f'Results/SoCEV_Heatmaps_All_EVs.png', dpi=300, bbox_inches='tight')
plt.close()
# Sum all the PEV values for each time step (Energy Supplied by all EVs)
sum_PEV_values_c = [
    sum(model.PEV_c[ev, t].value for ev in Ωev) for t in Ωt
]  # Sum of PEV values for each time step

sum_PEV_values_d = [
    sum(-1*model.PEV_d[ev, t].value for ev in Ωev) for t in Ωt
]  # Sum of PEV values for each time step
# Create the figure for plotting all variables

counter = 0
for ev in Ωev:
    for t in Ωt:
        if model.PEV_c[ev, t].value > 0 and model.PEV_d[ev, t].value > 0:
            print(f"Erro: EV {ev} está carregando e descarregando ao mesmo tempo as {t}")
            counter += 1
print(f"Total de {counter} erros")

fig = go.Figure()

# Add trace for summed Energy Supplied by EVs (PEV)
fig.add_trace(go.Scatter(
    x=Ωt,
    y=sum_PEV_values_c,
    mode='lines+markers',
    name='EV charging power (kW)',
    line=dict(color='green')
))


# Add trace for summed Energy Supplied by EVs (PEV)
fig.add_trace(go.Scatter(
    x=Ωt,
    y=sum_PEV_values_d,
    mode='lines+markers',
    name='EV discharge power (kW)',
    line=dict(color='blue')
))
# Add trace for Energy Bought (PS)
PSp_values = [model.PSp[t].value - model.PSn[t].value for t in Ωt]  # Extract energy bought for each time step
fig.add_trace(go.Scatter(
    x=Ωt,
    y=PS_values,
    mode='lines+markers',
    name='EDS power (kW)',
    line=dict(color='red')
))


# Add trace for Solar Generation (PV)
PV_generated = [model.PPVmax.value * fs['5']['pv'][t] for t in Ωt]  # Extract solar generation for each time step
fig.add_trace(go.Scatter(
    x=Ωt,
    y=PV_generated,
    mode='lines+markers',
    name='Solar Generation (kWh)',
    line=dict(color='orange')
))

# Add trace for Demand
demand_values = [data["EDS"]["LOAD"] * fs['5']['load'][t] for t in Ωt]  # Extract demand for each time step
fig.add_trace(go.Scatter(
    x=Ωt,
    y=demand_values,
    mode='lines+markers',
    name='Demand (kWh)',
    line=dict(color='black', dash='dot')
))

# Update layout of the figure
fig.update_layout(
    title="Energy Supplied by EVs (PEV), Energy Bought (PS), Solar Generation, and Demand Over Time",
    xaxis_title="Time",
    yaxis_title="Value (kWh)",
    legend_title="Variables",
    template="plotly",
    showlegend=True
)

# Show the figure
fig.show()

# Assuming Ωev and Ωt are the sets of EVs and time periods, respectively
# Extract αEV[ev] for each EV and time period and store in a dictionary
alpha_values = {}

# Loop through all EVs and time periods and extract the αEV[ev] values
for ev in Ωev:
    alpha_values[ev] = [model.αEV[ev, t].value for t in Ωt]

# Convert the dictionary into a DataFrame for easier plotting with Plotly
df_alpha = pd.DataFrame(alpha_values, index=[t for t in Ωt])

# Create the heatmap using Plotly
fig = go.Figure(data=go.Heatmap(
    z=df_alpha.values,               # Values to plot
    x=df_alpha.columns,               # Time periods (Ωt)
    y=df_alpha.index,                 # EVs (Ωev)
    colorscale='YlGnBu',              # Color scale
    colorbar=dict(title='Charging (1 = Yes, 0 = No)'), # Colorbar with label
))

# Update layout for better titles and axis labels
fig.update_layout(
    title="Charging Status (αEV) of Each EV Over Time",
    xaxis_title="Time",
    yaxis_title="EVs",
    xaxis=dict(tickangle=45),         # Rotate time axis for readability
    autosize=True,
)

# Show the plot in the browser (interactive)
fig.show()

# Optionally, you can save the plot as an HTML file to view later
fig.write_html("charging_status_heatmap.html")