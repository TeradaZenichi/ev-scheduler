from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os


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

Δt = data['timestep']/60

# Sets definition
# Ωs = list(fs.keys())
Ωs = data['scenarios']
Ωe = list(EV.keys())
Ωc = list()
Ωt = list()

Ωc_dc = dict()
Ωc_ac = dict()


t = datetime.strptime('00:00', "%H:%M")
while t < datetime.strptime('23:59', "%H:%M"):
    Ωt.append(t.strftime("%H:%M"))
    t += timedelta(hours=Δt)

# EVCSs definition
for evcs in EVCS.keys():
    dc = [connector for connector in EVCS[evcs]['connector'].keys() if EVCS[evcs]['connector'][connector]['current'] == 'DC']
    ac = [connector for connector in EVCS[evcs]['connector'].keys() if EVCS[evcs]['connector'][connector]['current'] == 'AC']
    Ωc_dc[evcs] = dc
    Ωc_ac[evcs] = ac
    for connector in EVCS[evcs]['connector'].keys():
        Ωc.append((evcs, connector))

# Probability definition
πs = {s: fs[s]['prob'] for s in Ωs}


# Variables definition
model = pyo.ConcreteModel(data['model name'])

model.Peds      = pyo.Var(Ωt, Ωs, domain=pyo.Reals, bounds=(data['EDS']['Pmin'], data['EDS']['Pmax']))
model.Ppeds     = pyo.Var(Ωt, Ωs, domain=pyo.NonNegativeReals)
model.Pneds     = pyo.Var(Ωt, Ωs, domain=pyo.NonNegativeReals)

model.Pbess_c   = pyo.Var(Ωt, Ωs, domain=pyo.NonNegativeReals, bounds=(0, data['BESS']['Pmax']))
model.Pbess_d   = pyo.Var(Ωt, Ωs, domain=pyo.NonNegativeReals, bounds=(0, data['BESS']['Pmax']))
model.SoCbess   = pyo.Var(Ωt, Ωs, domain=pyo.NonNegativeReals, bounds=(0,1))
model.γbess_c   = pyo.Var(Ωt, Ωs, domain=pyo.Binary)
model.γbess_d   = pyo.Var(Ωt, Ωs, domain=pyo.Binary)

model.SoCEV     = pyo.Var(Ωt, Ωs, Ωe, domain=pyo.NonNegativeReals, bounds=(0,1))
model.Pαc       = pyo.Var(Ωt, Ωs, Ωe, Ωc, domain=pyo.NonNegativeReals)
model.Pαd       = pyo.Var(Ωt, Ωs, Ωe, Ωc, domain=pyo.NonNegativeReals)
model.αEV       = pyo.Var(Ωt, Ωs, Ωe, Ωc, domain=pyo.Binary)
model.γαc       = pyo.Var(Ωt, Ωs, Ωe, Ωc, domain=pyo.Binary)
model.γαd       = pyo.Var(Ωt, Ωs, Ωe, Ωc, domain=pyo.Binary)


"""
    Objective function
"""
def objective(model):
    cEVc = data['cost']['EVc']
    cEVd = data['cost']['EVd']
    return sum(
        Δt * πs[s] * cost[t] * model.Ppeds[t, s] +  Δt * sum(
            cEVd * model.Pαd[t, s, e, evcs, connector] - cEVc * model.Pαc[t, s, e, evcs, connector]  
            for e in Ωe for evcs, connector in Ωc) for t in Ωt for s in Ωs
    )
model.objective = pyo.Objective(rule=objective, sense=pyo.minimize)

"""
    Power balancing and EDS constraints
"""
def powerflow(model, t, s):
    pv = data['PV']['Pmax'] * fs[s]['pv'][t]
    load = data['load']['Pmax'] * fs[s]['load'][t]
    return model.Peds[t, s] + pv + model.Pbess_d[t, s] == load + model.Pbess_c[t, s] + sum(
        model.Pαc[t, s, e, evcs, connector] - model.Pαd[t, s, e, evcs, connector]  
        for e in Ωe for evcs, connector in Ωc
    )
model.powerflow = pyo.Constraint(Ωt, Ωs, rule=powerflow)

def edsconstraint(model, t, s):
    return model.Ppeds[t, s] - model.Pneds[t, s] == model.Peds[t, s]
model.edsconstraint = pyo.Constraint(Ωt, Ωs, rule=edsconstraint)

"""
    BESS constraints
"""
def besssoc(model, t, s):
    Emax = data['BESS']['Emax']
    η = data['BESS']['efficiency']
    if t == Ωt[0]:
        return model.SoCbess[t, s] == data['BESS']['SoC'] + Δt * (η * model.Pbess_c[t, s] - model.Pbess_d[t, s] / η) * (1/Emax)
    else:
        t0 = (datetime.strptime(t, "%H:%M") - timedelta(hours=Δt)).strftime("%H:%M")
        return model.SoCbess[t, s] == model.SoCbess[t0, s] + Δt * (η * model.Pbess_c[t, s] - model.Pbess_d[t, s] / η) * (1/Emax)
model.besssoc = pyo.Constraint(Ωt, Ωs, rule=besssoc)

def besscharging(model, t, s):
    return model.Pbess_c[t, s] <= data['BESS']['Pmax'] * model.γbess_c[t, s]
model.besscharging = pyo.Constraint(Ωt, Ωs, rule=besscharging)

def bessdischarging(model, t, s):
    return model.Pbess_d[t, s] <= data['BESS']['Pmax'] * model.γbess_d[t, s]
model.bessdischarging = pyo.Constraint(Ωt, Ωs, rule=bessdischarging)

def besscondition(model, t, s):
    return model.γbess_c[t, s] + model.γbess_d[t, s] <= 1
model.besscondition = pyo.Constraint(Ωt, Ωs, rule=besscondition)

"""
    EVs charging and discharging constraints
"""
def evsocconstraint(model, t, s, e):
    if t == Ωt[0]:
        return model.SoCEV[t, s, e] == EV[e]['SoCini'] + Δt * sum(
            EVCS[evcs]['connector'][connector]["efficiency"] * model.Pαc[t, s, e, evcs, connector] -
            model.Pαd[t, s, e, evcs, connector] / EVCS[evcs]['connector'][connector]["efficiency"]
            for evcs, connector in Ωc
        ) * (1/EV[e]['Emax'])
    else:
        t0 = (datetime.strptime(t, "%H:%M") - timedelta(hours=Δt)).strftime("%H:%M")
        return model.SoCEV[t, s, e] == model.SoCEV[t0, s, e] + Δt * sum(
            EVCS[evcs]['connector'][connector]["efficiency"] * model.Pαc[t, s, e, evcs, connector] -
            model.Pαd[t, s, e, evcs, connector] / EVCS[evcs]['connector'][connector]["efficiency"]
            for evcs, connector in Ωc
        ) * (1/EV[e]['Emax'])
model.evsocconstraint = pyo.Constraint(Ωt, Ωs, Ωe, rule=evsocconstraint)

def evsocdefinition(model, t, s, e):
    if datetime.strptime(t, "%H:%M") < datetime.strptime(EV[e]['arrival'], "%H:%M"):
        return model.SoCEV[t, s, e] == EV[e]['SoCini']
    elif datetime.strptime(t, "%H:%M") >= datetime.strptime(EV[e]['departure'], "%H:%M"):
        return model.SoCEV[t, s, e] == 1
    else:
        return pyo.Constraint.Skip
model.evsocdefinition = pyo.Constraint(Ωt, Ωs, Ωe, rule=evsocdefinition)

def evcs_αconstraint(model, t, s, e):
    return sum(model.αEV[t, s, e, evcs, connector] for evcs, connector in Ωc) <= 1
model.evcs_αconstraint = pyo.Constraint(Ωt, Ωs, Ωe, rule=evcs_αconstraint)

def ev_αconstraint(model, t, s, evcs, connector):
    return sum(model.αEV[t, s, e, evcs, connector] for e in Ωe) <= 1
model.ev_αconstraint = pyo.Constraint(Ωt, Ωs, Ωc, rule=ev_αconstraint)

def ev_lockconstraint(model, t, s, e, evcs, connector):
    if datetime.strptime(t, "%H:%M") < datetime.strptime(EV[e]['arrival'], "%H:%M"):
        return model.αEV[t, s, e, evcs, connector] == 0
    elif datetime.strptime(t, "%H:%M") == datetime.strptime(EV[e]['arrival'], "%H:%M"):
        return sum(model.αEV[t, s, e, evcs, connector] for evcs, connector in Ωc) == 1
    elif datetime.strptime(t, "%H:%M") > datetime.strptime(EV[e]['departure'], "%H:%M"):
        return model.αEV[t, s, e, evcs, connector] == 0
    else:
        t0 = (datetime.strptime(t, "%H:%M") - timedelta(hours=Δt)).strftime("%H:%M")
        return model.αEV[t, s, e, evcs, connector] == model.αEV[t0, s, e, evcs, connector]
model.ev_lockconstraint = pyo.Constraint(Ωt, Ωs, Ωe, Ωc, rule=ev_lockconstraint)


def αchargingconstraint(model, t, s, e, evcs, connector):
    return model.Pαc[t, s, e, evcs, connector] <= EVCS[evcs]['connector'][connector]['Pmaxc'] * model.αEV[t, s, e, evcs, connector]
model.αchargingconstraint = pyo.Constraint(Ωt, Ωs, Ωe, Ωc, rule=αchargingconstraint)

def αdischargingconstraint(model, t, s, e, evcs, connector):
    return model.Pαd[t, s, e, evcs, connector] <= EVCS[evcs]['connector'][connector]['Pmaxd'] * model.αEV[t, s, e, evcs, connector]
model.αdischargingconstraint = pyo.Constraint(Ωt, Ωs, Ωe, Ωc, rule=αdischargingconstraint)

def αγcharging(model, t, s, e, evcs, connector):
    return model.Pαc[t, s, e, evcs, connector] <= EVCS[evcs]['connector'][connector]['Pmaxc'] * model.γαc[t, s, e, evcs, connector]
model.αγcharging = pyo.Constraint(Ωt, Ωs, Ωe, Ωc, rule=αγcharging)

def αγdischarging(model, t, s, e, evcs, connector):
    return model.Pαd[t, s, e, evcs, connector] <= EVCS[evcs]['connector'][connector]['Pmaxd'] * model.γαd[t, s, e, evcs, connector]
model.αγdischarging = pyo.Constraint(Ωt, Ωs, Ωe, Ωc, rule=αγdischarging)

def αγcondition(model, t, s, e, evcs, connector):
    return model.γαc[t, s, e, evcs, connector] + model.γαd[t, s, e, evcs, connector] <= 1
model.αγcondition = pyo.Constraint(Ωt, Ωs, Ωe, Ωc, rule=αγcondition)

def EVchargingarrival(model, t, s, e):
    if datetime.strptime(t, "%H:%M") < datetime.strptime(EV[e]['arrival'], "%H:%M"):
        return model.Pαc[t, s, e, evcs, connector] == 0
    else:
        return pyo.Constraint.Skip
model.EVchargingarrival = pyo.Constraint(Ωt, Ωs, Ωe, rule=EVchargingarrival)

def EVchargingdeparture(model, t, s, e):
    if datetime.strptime(t, "%H:%M") > datetime.strptime(EV[e]['departure'], "%H:%M"):
        return model.Pαc[t, s, e, evcs, connector] == 0
    else:
        return pyo.Constraint.Skip
model.EVchargingdeparture = pyo.Constraint(Ωt, Ωs, Ωe, rule=EVchargingdeparture)

def EVdischargingarrival(model, t, s, e):
    if datetime.strptime(t, "%H:%M") < datetime.strptime(EV[e]['arrival'], "%H:%M"):
        return model.Pαd[t, s, e, evcs, connector] == 0
    else:
        return pyo.Constraint.Skip
model.EVdischargingarrival = pyo.Constraint(Ωt, Ωs, Ωe, rule=EVdischargingarrival)

def EVdischargingdeparture(model, t, s, e):
    if datetime.strptime(t, "%H:%M") > datetime.strptime(EV[e]['departure'], "%H:%M"):
        return model.Pαd[t, s, e, evcs, connector] == 0
    else:
        return pyo.Constraint.Skip
model.EVdischargingdeparture = pyo.Constraint(Ωt, Ωs, Ωe, rule=EVdischargingdeparture)

def αDCconstraint(model, t, s, evcs):
    if len(Ωc_dc[evcs]) > 1:
        return sum(model.αEV[t, s, e, evcs, connector] for e in Ωe for connector in Ωc_dc[evcs]) <= EVCS[evcs]['NDC']
    else:
        return pyo.Constraint.Skip
model.αDCconstraint = pyo.Constraint(Ωt, Ωs, Ωc_dc.keys(), rule=αDCconstraint)

def αACconstraint(model, t, s, evcs):
    if len(Ωc_ac[evcs]) > 1:    
        return sum(model.αEV[t, s, e, evcs, connector] for e in Ωe for connector in Ωc_ac[evcs]) <= EVCS[evcs]['NAC']
    else:
        return pyo.Constraint.Skip        
model.αACconstraint = pyo.Constraint(Ωt, Ωs, Ωc_ac.keys(), rule=αACconstraint)

def EVconnectorconstraint(model, t, s, e, evcs, connector):
    if EVCS[evcs]['connector'][connector]['type'] != EV[e]['Connector']:
        return model.αEV[t, s, e, evcs, connector] == 0
    else:
        return pyo.Constraint.Skip
model.EVconnectorconstraint = pyo.Constraint(Ωt, Ωs, Ωe, Ωc, rule=EVconnectorconstraint)

results = SolverFactory('gurobi').solve(model)

print(results.solver.status)

if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
    print("Optimal solution found")
    print(pyo.value(model.objective))


if not os.path.exists('Results'):
    os.makedirs('Results')

if not os.path.exists('Results/scenarios'):
    os.makedirs('Results/scenarios')

mpl.rc('font',family = 'serif', serif = 'cmr10')
plt.rcParams['axes.unicode_minus'] = False




for s in Ωs:

    plt.figure()
    plt.plot(Ωt, [pyo.value(model.Peds[t, s]) for t in Ωt], label='EDS', color='red')
    plt.plot(Ωt, [-data['PV']['Pmax'] * fs[s]['pv'][t] for t in Ωt], label='PV', color='orange')
    plt.plot(Ωt, [data['load']['Pmax'] * fs[s]['load'][t] for t in Ωt], label='Load', color='black', marker='o', linestyle='dashed', markersize=3)
    plt.bar(Ωt, [pyo.value(sum(model.Pαc[t, s, e, evcs, connector] - model.Pαd[t, s, e, evcs, connector] for e in Ωe for evcs, connector in Ωc if evcs != "EVCS4")) for t in Ωt], label='EVCSS (123)', color='#00c8ff', alpha=0.5)
    plt.bar(Ωt, [pyo.value(sum(model.Pαc[t, s, e, evcs, connector] - model.Pαd[t, s, e, evcs, connector] for e in Ωe for evcs, connector in Ωc if evcs == "EVCS4")) for t in Ωt], label='V2G', color='purple', alpha=0.5)
    plt.bar(Ωt, [pyo.value(model.Pbess_c[t, s]) - pyo.value(model.Pbess_d[t, s] ) for t in Ωt], label='BESS', color='green', alpha=0.5)
    plt.ylabel('Power [kW]')
    plt.xlabel('Timestamp')
    plt.ylim(-40, 80)
    plt.xticks(Ωt[::int(len(Ωt)/12)], rotation=90)  # Set x-axis ticks at every nth element of Ωt with rotation
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', ncol = 3,fontsize='small')
    plt.savefig(f'Results/Operation_{s}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Results/Operation_{s}.pdf', dpi=300, bbox_inches='tight')
    plt.close('All')


for s in Ωs:
    plt.figure()
    plt.plot(Ωt, [pyo.value(model.SoCbess[t, s]) for t in Ωt], label='SoC BESS', color='blue')
    for e in Ωe:
        plt.plot(Ωt, [pyo.value(model.SoCEV[t, s, e]) for t in Ωt], label=f'SoC EV {e}', linestyle='dashed', marker='o', markersize=1)
    plt.ylabel('State of Charge')
    plt.xlabel('Timestamp')
    plt.xticks(Ωt[::int(len(Ωt)/6)])  # Set x-axis ticks at every 4th element of Ωt
    plt.legend(loc='upper right')
    plt.savefig(f'Results/SoC_{s}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Results/SoC_{s}.pdf', dpi=300, bbox_inches='tight')
    plt.close('All')

for s in Ωs:
    dic_values = {}  # Para os valores numéricos (idev)
    dic_annotations = {}  # Para as anotações

    for evcs, connector in Ωc:
        values = []  # Lista para armazenar os valores de idev
        annotations = []  # Lista para armazenar as anotações

        for t in Ωt:
            idev = 0
            idsoc = 0
            annotation = '0\n0'  # Anotação padrão se nenhum EV estiver conectado

            for e in Ωe:
                if pyo.value(model.αEV[t, s, e, evcs, connector]) == 1:
                    if idev > 0:
                        print(f'Error: More than one EV connected to {evcs}{connector} at {t}')
                    idev = int(e)
                    idsoc = int(100 * pyo.value(model.SoCEV[t, s, e]))
                    annotation = f'{idev}\n{idsoc}%'  # Atualiza a anotação com valores reais
            if idev == 0:
                annotation = f'\n'  # Atualiza a anotação com valores reais

            values.append(idev)  # Adiciona o valor de idev para a coloração
            annotations.append(annotation)  # Adiciona a string formatada para anotações

        dic_values[f'{evcs}{connector}'] = values
        dic_annotations[f'{evcs}{connector}'] = annotations

    df_values = pd.DataFrame.from_dict(dic_values, orient='index', columns=pd.to_datetime(Ωt, format='%H:%M').time)
    df_annotations = pd.DataFrame.from_dict(dic_annotations, orient='index', columns=pd.to_datetime(Ωt, format='%H:%M').time)

    plt.figure(figsize=(16, 4))
    ax = sns.heatmap(df_values, cmap='tab20', cbar=False, linewidths=.5)

    # Adiciona anotações manualmente
    for y, row in enumerate(df_annotations.values):
        for x, cell in enumerate(row):
            idev, idsoc = cell.split('\n')
            ax.text(x + 0.5, y + 0.3, idev, ha='center', va='center', fontsize=6)  # Tamanho da fonte para idev
            ax.text(x + 0.5, y + 0.7, idsoc, ha='center', va='center', fontsize=4, color='black', rotation=-90)  # Tamanho da fonte para idsoc, ajustado e em cinza

    plt.title(f'EVCSs - Scenario {s}')
    plt.xlabel('Timestamp')
    plt.ylabel('EVCSs')
    plt.savefig(f'Results/EVCS-s{s}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Results/EVCS-s{s}.pdf', dpi=300, bbox_inches='tight')    
    plt.close('All')



a = 1