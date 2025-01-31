import plotly.graph_objects as go
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, Reals, SolverFactory, Binary
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

# Define contingency start times as valid timestamps from Ωt
Ωc = [100, 48, 38, 65]  # Example: Contingency starts at 02:00 (index 8) and 03:45 (index 15)

# Define contingency duration in timesteps (2 hours if Δt = 15 minutes)
contingency_duration = 8  # 2 hours = 8 timesteps (15 minutes each)

# Add contingency probabilities (if applicable)
πc = {c: 0.25 for c in Ωc}  # Equal probability for each contingency scenario

t = datetime.strptime('00:00', "%H:%M")
while t < datetime.strptime('23:59', "%H:%M"):
    Ωt.append(t.strftime("%H:%M"))
    t += timedelta(minutes=Δt)


# Definindo o modelo
model = ConcreteModel()


# Variable definition with contingency scenarios
model.PS = Var(Ωt, Ωc, domain=Reals, bounds=(data["EDS"]['Pmin'], data["EDS"]['Pmax']))  # Substation power
model.PSp = Var(Ωt, Ωc, domain=NonNegativeReals)  # Energy bought from the grid
model.PSn = Var(Ωt, Ωc, domain=NonNegativeReals)  # Energy sold to the grid

# BESS variables
model.PBESS_c = Var(Ωt, Ωc, within=NonNegativeReals)  # BESS charging power
model.PBESS_d = Var(Ωt, Ωc, within=NonNegativeReals)  # BESS discharging power
model.SoCBESS = Var(Ωt, Ωc, within=NonNegativeReals)  # BESS State of Charge
model.BESS_CAPEX = Var(within=NonNegativeReals)  # BESS CAPEX
model.EmaxBESS = Var(within=NonNegativeReals)  # BESS maximum energy capacity

# EV operation variables
model.SoCEV = Var(Ωev, Ωt, Ωc, domain=NonNegativeReals)  # EV State of Charge
model.PEV_c = Var(Ωev, Ωt, Ωc, domain=NonNegativeReals)  # EV charging power
model.PEV_d = Var(Ωev, Ωt, Ωc, domain=NonNegativeReals)  # EV discharging power
model.αEV = Var(Ωev, Ωt, Ωc, domain=Binary)  # EV charging factor

# Thermal generator variables
model.TG_MAX_CAP = Var(within=NonNegativeReals)  # Maximum installed capacity (affects CAPEX)
model.PTG = Var(Ωt, Ωc, domain=NonNegativeReals)  # Power output (affects OPEX)
model.TG_CAPEX = Var(within=NonNegativeReals)  # Fixed cost for capacity (based on TG_MAX_CAP)
model.TG_OPEX = Var(within=NonNegativeReals)  # Operational cost for usage (based on PTG)

# PV variable
model.PPVmax = Var(domain=NonNegativeReals)  # Maximum PV generation

# Economic variables
model.CAPEX = Var(within=NonNegativeReals)  # Total CAPEX
model.OPEX = Var(within=NonNegativeReals)  # Total OPEX


# Add operational cost parameters (define these in your data or parameters.json)
data["BESS_charge_cost"] = 0.05  # Cost per kWh for BESS charging
data["BESS_discharge_cost"] = 0.03  # Cost per kWh for BESS discharging
data["EV_discharge_cost"] = 0.10  # Cost per kWh for EV discharging


def objective_rule(model):
    # CAPEX terms
    total_capex = model.CAPEX + model.TG_CAPEX  # Include thermal generator CAPEX

    # OPEX terms (with interest rate applied over the years)
    total_opex = 0
    for n in range(1, data["OPEX"]["years"] + 1):
        total_opex += model.OPEX * 365 * ((1 + data["OPEX"]["rate"]) ** (n - 1))

    # Total cost
    total_cost = total_capex + total_opex

    return total_cost

model.objective = Objective(rule=objective_rule, sense=pyo.minimize)

# Contingency constraint: Limit substation power to zero during contingency periods
def substation_contingency_rule(model, t, c):
    """Limit substation power to zero during contingency periods."""
    t_index = Ωt.index(t)
    
    if c <= t_index < c + contingency_duration:
        return model.PS[t, c] == 0  # Substation power is zero during contingency
    else:
        return Constraint.Skip  # No constraint outside contingency periods

model.substation_contingency = Constraint(Ωt, Ωc, rule=substation_contingency_rule)


# Thermal generator OPEX constraint
def tg_opex_rule(model):
    return model.TG_OPEX == sum(
        data["TG"]["OPEX"] * model.PTG[t, c] * (Δt / 60)  # Operational cost per kWh
        for t in Ωt for c in Ωc
    )

model.tg_opex_constraint = Constraint(rule=tg_opex_rule)

# Thermal generator power output limits
def tg_power_rule(model, t, c):
    return model.PTG[t, c] <= model.TG_MAX_CAP  # Power output cannot exceed maximum installed capacity

model.tg_power_constraint = Constraint(Ωt, Ωc, rule=tg_power_rule)


# CAPEX constraint (ensuring CAPEX is the sum of InstPV and InstWT)
def capex_constraint_rule(model):
    PV = model.PPVmax * data["CAPEX"]['PV']  # PV installation cost
    BESS_CAPEX = model.EmaxBESS * data["CAPEX"]["BESS"]
    TG = data["TG"]["CAPEX"] * model.TG_MAX_CAP
    return model.CAPEX == PV + BESS_CAPEX + TG
model.capex_constraint = Constraint(rule=capex_constraint_rule)

def opex_constraint_rule(model):
    # Substation OPEX (cost of buying power from the grid)
    substation_opex = sum(cost[t] * model.PSp[t, c] for t in Ωt for c in Ωc)

    # BESS OPEX (charging and discharging costs)
    bess_opex = sum(
        data["BESS_charge_cost"] * model.PBESS_c[t, c] * (Δt / 60) +
        data["BESS_discharge_cost"] * model.PBESS_d[t, c] * (Δt / 60)
        for t in Ωt for c in Ωc
    )

    # EV OPEX (discharging costs)
    ev_opex = sum(
        data["EV_discharge_cost"] * model.PEV_d[ev, t, c] * (Δt / 60)
        for ev in Ωev for t in Ωt for c in Ωc
    )

    # Thermal generator OPEX (operational cost)
    tg_opex = sum(
        data["TG"]["OPEX"] * model.PTG[t, c] * (Δt / 60)
        for t in Ωt for c in Ωc
    )

    # Total OPEX
    return model.OPEX == substation_opex + bess_opex + ev_opex + tg_opex

model.opex_constraint = Constraint(rule=opex_constraint_rule)

def eds_power_rule(model, t, c):
    return model.PS[t, c] == model.PSp[t, c] - model.PSn[t, c]
model.eds_power_constraint = Constraint(Ωt, Ωc, rule=eds_power_rule)

# Define the rule for the PSp (charging power) constraint
def psp_constraint(model, t, c):
    return model.PSp[t, c] <= data["EDS"]['Pmax']

model.psp_constraint = Constraint(Ωt, Ωc, rule=psp_constraint)

# Capacidade máxima da bateria e taxa de carga/descarregamento
crate = data["BESS"]['crate']  # Taxa de carga/descarregamento do BESS, a partir do arquivo

# Atualizando as funções para o modelo de BESS

def bess_power_rule_c(model, t, c):
    return model.PBESS_c[t, c] <= model.EmaxBESS * data["BESS"]['crate']

model.bess_power_c = Constraint(Ωt, Ωc, rule=bess_power_rule_c)

def bess_power_rule_d(model, t, c):
    return model.PBESS_d[t, c] <= model.EmaxBESS * data["BESS"]['crate']

model.bess_power_d = Constraint(Ωt, Ωc, rule=bess_power_rule_d)

def soceb_update(model, t, c):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    η = 1  # BESS efficiency
    return model.SoCBESS[t2, c] == model.SoCBESS[t1, c] + η * model.PBESS_c[t2, c] * (Δt / 60) - model.PBESS_d[t2, c] * (η * Δt / 60)

model.soceb_update = Constraint(Ωt, Ωc, rule=soceb_update)



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

def ev_power_rule_c(model, ev, t, c):
    return model.PEV_c[ev, t, c] <= model.αEV[ev, t, c] * EV[ev]['Emax']

model.ev_power_c = Constraint(Ωev, Ωt, Ωc, rule=ev_power_rule_c)

def ev_power_rule_d(model, ev, t, c):
    return model.PEV_d[ev, t, c] <= model.αEV[ev, t, c] * EV[ev]['Emax']

model.ev_power_d = Constraint(Ωev, Ωt, Ωc, rule=ev_power_rule_d)

def socev_update(model, ev, t, c):
    ta = datetime.strptime(EV[ev]['arrival'], "%H:%M")
    td = datetime.strptime(EV[ev]['departure'], "%H:%M")
    t0 = datetime.strptime(t, "%H:%M")
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    η = EV[ev]['eff']
    if ta < td and t0 > ta and t0 <= td:
        return model.SoCEV[ev, t2, c] == model.SoCEV[ev, t1, c] + η * model.PEV_c[ev, t2, c] * (Δt / 60) - model.PEV_d[ev, t2, c] * (η * Δt / 60)
    elif ta > td and not (t0 > td and t0 <= ta):
        return model.SoCEV[ev, t2, c] == model.SoCEV[ev, t1, c] + η * model.PEV_c[ev, t2, c] * (Δt / 60) - model.PEV_d[ev, t2, c] * (η * Δt / 60)
    elif t0 == ta:
        return model.SoCEV[ev, t, c] == EV[ev]['SoCini'] * EV[ev]['Emax']
    else:
        return model.SoCEV[ev, t, c] == 0

model.socev_update = Constraint(Ωev, Ωt, Ωc, rule=socev_update)



# Modified version of the V2G constraints that handles time correctly
def v2g_constraints(model, ev, t, c):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")   
    η = EV[ev]['eff']
    
    return model.PEV_c[ev, t2, c] <= (EV[ev]['Emax'] - model.SoCEV[ev, t1, c])  / (η * Δt / 60) 

model.v2g_charge = Constraint(Ωev, Ωt, Ωc, rule=v2g_constraints)

def v2g_discharge_constraints(model, ev, t, c):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")
    η = EV[ev]['eff']

    return model.PEV_d[ev, t2, c] <= (model.SoCEV[ev, t1, c] * η) / (Δt / 60)

model.v2g_discharge = Constraint(Ωev, Ωt, Ωc, rule=v2g_discharge_constraints)

def v2g_discharge_max(model, ev, t, c):
    t2 = datetime.strptime(t, "%H:%M").strftime("%H:%M")
    t1 = (datetime.strptime(t, "%H:%M") - timedelta(minutes=Δt)).strftime("%H:%M")

    # Discharge power should not exceed the calculated max value
    return model.PEV_d[ev, t2, c] <= EV[ev]['Pmax_d'] - (EV[ev]['Pmax_d'] / EV[ev]['Pmax_c']) * model.PEV_c[ev, t2, c]

model.v2g_discharge_max = Constraint(Ωev, Ωt, Ωc, rule=v2g_discharge_max)


def EV_departure_rule(model, ev, c):
    return model.SoCEV[ev, EV[ev]['departure'], c] == EV[ev]['Emax']

model.EV_departure = Constraint(Ωev, Ωc, rule=EV_departure_rule)


# Update power balance constraint
def balanco_carga(model, t, c):
    pv = model.PPVmax * fs['5']['pv'][t]  # PV generation
    load = data["EDS"]["LOAD"] * fs['5']['load'][t]  # Demand
    return (
        model.PS[t, c] + pv + sum(model.PEV_d[ev, t, c] for ev in Ωev) + model.PBESS_d[t, c] + model.PTG[t, c] ==
        sum(model.PEV_c[ev, t, c] for ev in Ωev) + model.PBESS_c[t, c] + load
    )

model.balanco_carga = Constraint(Ωt, Ωc, rule=balanco_carga)



# Solve the model
results = SolverFactory('gurobi').solve(model)

# Extract results for each contingency scenario
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
        'PTG': [model.PTG[t, c].value for t in Ωt]  # Add thermal generator power output
    }
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

import plotly.graph_objects as go

# Define a professional color palette (colorblind-friendly)
colors = {
    'Substation': 'red',
    'Demand': 'black',
    'Solar': 'orange',
    'Net EV Power': 'purple',
    'Net BESS Power': 'blue',
    'Thermal Generator': 'green'  # Add a color for the thermal generator
}

# Generate graphs for each contingency scenario
for c in Ωc:
    # Extract data for the current contingency scenario
    PS_values = contingency_results[c]['PS']
    PBESS_c_values = contingency_results[c]['PBESS_c']
    PBESS_d_values = contingency_results[c]['PBESS_d']
    PEV_c_values = contingency_results[c]['PEV_c']
    PEV_d_values = contingency_results[c]['PEV_d']
    PV_generated = contingency_results[c]['PV']
    demand_values = contingency_results[c]['Demand']
    PTG_values = contingency_results[c]['PTG']  # Thermal generator power output

    # Calculate net EV and BESS power
    net_EV_power = [PEV_c_values[t_idx] - PEV_d_values[t_idx] for t_idx in range(len(Ωt))]
    net_BESS_power = [PBESS_c_values[t_idx] - PBESS_d_values[t_idx] for t_idx in range(len(Ωt))]

    # Create the figure
    fig = go.Figure()

    # Add traces for each component
    fig.add_trace(go.Scatter(
        x=Ωt, 
        y=PS_values, 
        mode='lines', 
        name='Substation Power (PS)', 
        line=dict(color=colors['Substation'], width=2)
    ))

    fig.add_trace(go.Scatter(
        x=Ωt, 
        y=demand_values, 
        mode='lines', 
        name='Demand', 
        line=dict(color=colors['Demand'], width=2, dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=Ωt, 
        y=PV_generated, 
        mode='lines', 
        name='Solar Generation', 
        line=dict(color=colors['Solar'], width=2)
    ))

    fig.add_trace(go.Scatter(
        x=Ωt, 
        y=net_EV_power, 
        mode='lines', 
        name='Net EV Power (EVc - EVd)', 
        line=dict(color=colors['Net EV Power'], width=2)
    ))

    fig.add_trace(go.Scatter(
        x=Ωt, 
        y=net_BESS_power, 
        mode='lines', 
        name='Net BESS Power (BESSc - BESSd)', 
        line=dict(color=colors['Net BESS Power'], width=2)
    ))

    # Add thermal generator trace
    fig.add_trace(go.Scatter(
        x=Ωt, 
        y=PTG_values, 
        mode='lines', 
        name='Thermal Generator Power (PTG)', 
        line=dict(color=colors['Thermal Generator'], width=2)
    ))

    # Update layout for a professional look
    fig.update_layout(
        title=dict(
            text=f"Power in the System Over Time (Contingency {c})",
            x=0.5,  # Center the title
            font=dict(size=18, family='Times New Roman')  # Professional font
        ),
        xaxis=dict(
            title="Time (HH:MM)",
            title_font=dict(size=14, family='Times New Roman'),
            tickfont=dict(size=12, family='Times New Roman'),
            gridcolor='lightgray',  # Add grid lines
            showgrid=True
        ),
        yaxis=dict(
            title="Power (kW)",
            title_font=dict(size=14, family='Times New Roman'),
            tickfont=dict(size=12, family='Times New Roman'),
            gridcolor='lightgray',  # Add grid lines
            showgrid=True
        ),
        legend=dict(
            x=0.02,  # Position the legend
            y=0.98,
            font=dict(size=12, family='Times New Roman'),
            bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent background
            bordercolor='black',
            borderwidth=1
        ),
        plot_bgcolor='white',  # White background
        paper_bgcolor='white',  # White paper background
        margin=dict(l=80, r=40, b=80, t=80),  # Adjust margins
        hovermode='x unified'  # Show hover information for all traces
    )

    # Show the figure
    fig.show()

    # Save the figure as an HTML file
    fig.write_html(f"Results/Contingency_{c}_Power_Graph.html")

# # Create a figure with subplots to plot all the heatmaps
# fig, axes = plt.subplots(len(Ωev), 1, figsize=(16, 4 * len(Ωev)))

# # Loop through each Electric Vehicle (EV) for SoCEV heatmap plotting
# for ev_idx, (ev, ax) in enumerate(zip(Ωev, axes)):
#     dic_values = {}  # To store numerical values (SoC of EV)
#     dic_annotations = {}  # To store annotations with EV id and SoC percentage

#     # We will now loop through time steps (Ωt) for each EV
#     values = []  # List to store SoC values for each time step
#     annotations = []  # List to store annotations

#     for t in Ωt:  # Loop over time steps
#         # Fix the contingency scenario to the first one (c = Ωc[0])
#         c = Ωc[0]  # Use the first contingency scenario
#         # Get the SoC of the EV at time t and contingency c
#         soc = pyo.value(model.SoCEV[ev, t, c]) / EV[ev]['Emax']  # Correctly referencing SoCEV[ev, t, c]
#         annotation = f'{ev}\n{int(soc * 100)}%'  # Annotation with EV id and SoC percentage
#         values.append(soc)  # Append the SoC value
#         annotations.append(annotation)  # Append the annotation

#     dic_values[f'EV {ev}'] = values  # Store SoC values for the EV
#     dic_annotations[f'EV {ev}'] = annotations  # Store annotations for the EV

#     # Create a DataFrame for SoC values
#     df_values = pd.DataFrame.from_dict(dic_values, orient='index', columns=pd.to_datetime(Ωt, format='%H:%M').time)
    
#     # Create a DataFrame for annotations
#     df_annotations = pd.DataFrame.from_dict(dic_annotations, orient='index', columns=pd.to_datetime(Ωt, format='%H:%M').time)

#     # Plotting the heatmap
#     sns.heatmap(df_values, cmap='coolwarm', cbar_kws={'label': 'State of Charge (SoC)'},
#                 linewidths=.5, annot=df_annotations, fmt='', annot_kws={'size': 8}, ax=ax)

#     # Annotating with EV id and SoC percentage
#     for y, row in enumerate(df_annotations.values):
#         for x, cell in enumerate(row):
#             ev_id, soc_percent = cell.split('\n')
#             ax.text(x + 0.5, y + 0.3, ev_id, ha='center', va='center', fontsize=8, color='black')  # EV id
#             ax.text(x + 0.5, y + 0.7, soc_percent, ha='center', va='center', fontsize=6, color='gray')  # SoC percentage

#     ax.set_title(f'State of Charge (SoCEV) - EV {ev}')
#     ax.set_xlabel('Timestamp')
#     ax.set_ylabel('Electric Vehicle (EV)')
#     ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability
#     ax.tick_params(axis='y', rotation=0)  # Keep y-axis labels horizontal

# # Adjust layout for tight fitting
# plt.tight_layout()

# # Save all heatmaps into a single PNG
# plt.savefig(f'Results/SoCEV_Heatmaps_All_EVs.png', dpi=300, bbox_inches='tight')
# plt.close()

# Extract EV SoC values for a specific contingency scenario (e.g., the first one)
c = Ωc[0]  # Fix the contingency scenario to the first one in the list
ev_soc_values = {
    ev: [model.SoCEV[ev, t, c].value / EV[ev]['Emax'] for t in Ωt] for ev in Ωev
}

# Create DataFrame for EV SoC
df_ev_soc = pd.DataFrame(ev_soc_values, index=Ωt)

# Define a more distinct and visually appealing colorscale
colorscale = [
    [0.0, 'lightyellow'],  # Light yellow for low SoC
    [0.5, 'orange'],       # Orange for medium SoC
    [1.0, 'darkred']       # Dark red for high SoC
]

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=df_ev_soc.values.T,  # Transpose to match EVs on y-axis
    x=df_ev_soc.index,     # Timestamps on x-axis
    y=df_ev_soc.columns,   # EVs on y-axis
    colorscale=colorscale, # New colorscale for better visibility
    colorbar=dict(title='State of Charge (SoC)'),
    zmin=0,  # Set the minimum value for the colorscale
    zmax=1   # Set the maximum value for the colorscale
))

# Update layout
fig.update_layout(
    title="EV State of Charge (SoC) Over Time",
    xaxis_title="Time",
    yaxis_title="EVs",
    xaxis=dict(
        tickangle=45,  # Rotate x-axis labels for better readability
        tickmode='array',  # Use custom tick positions
        tickvals=df_ev_soc.index[::2],  # Show every 2nd timestamp to reduce clutter
        ticktext=df_ev_soc.index[::2],  # Corresponding labels
        tickfont=dict(size=12)  # Increase font size of timestamps
    ),
    yaxis=dict(
        tickfont=dict(size=12)  # Increase font size of EV labels
    ),
    autosize=True,
    margin=dict(l=100, r=100, b=100, t=100),  # Adjust margins to ensure full coverage
    plot_bgcolor='white'  # Set the plot background to white
)

# Make the bars clearer by increasing gaps between cells
fig.update_traces(
    xgap=3,  # Increase horizontal gap between bars
    ygap=3   # Increase vertical gap between bars
)

# Show the figure
fig.show()