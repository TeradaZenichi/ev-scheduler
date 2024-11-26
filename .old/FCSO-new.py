from pyomo.environ import *

model = ConcreteModel()

# Sets
model.ob = Set()  # Set of nodes
model.ol = Set(within=model.ob * model.ob)  # Set of circuits
model.od = Set()  # Set of demand levels

# Parameters
model.Tb = Param(model.ob)  # Node type: 0 = load, 1 = substation
model.PD = Param(model.ob, model.od)  # Active power demand at node i
model.QD = Param(model.ob, model.od)  # Reactive power demand at node i
model.R = Param(model.ol)  # Resistance on circuit ij
model.X = Param(model.ol)  # Reactance on circuit ij
model.Z2 = Param(model.ol)  # Square of the impedance on circuit ij
model.Imax = Param(model.ol)  # Maximum current limit on circuit ij
model.alpha = Param(model.od)  # Number of hours at demand level d
model.cls = Param(model.od)  # Energy loss cost at demand level d
model.Vnom = Param()  # Nominal voltage magnitude

# Variables
model.Vqdr = Var(model.ob, model.od, domain=NonNegativeReals)  # Squared voltage V[i,d]
model.PS = Var(model.ob, model.od)  # Active power supplied by the substation at node i
model.QS = Var(model.ob, model.od)  # Reactive power supplied by the substation at node i
model.Iqdr = Var(model.ol, model.od, domain=NonNegativeReals)  # Squared current I[i,j,d]
model.P = Var(model.ol, model.od)  # Active power flow on circuit ij
model.Q = Var(model.ol, model.od)  # Reactive power flow on circuit ij

# Objective function
def total_cost(model):
    return sum(model.cls[d] * model.alpha[d] * sum(model.R[i, j] * model.Iqdr[i, j, d] for i, j in model.ol) for d in model.od)

model.costObjective = Objective(rule=total_cost, sense=minimize)

# Constraints
# Active power balance
def active_power_balance(model, i, d):
    return (sum(model.P[j, i, d] for j, i in model.ol if (j, i) in model.ol) - 
            sum(model.P[i, j, d] + model.R[i, j] * model.Iqdr[i, j, d] for i, j in model.ol if (i, j) in model.ol) +
            model.PS[i, d] == model.PD[i, d])

model.activePowerBalance = Constraint(model.ob, model.od, rule=active_power_balance)

# Reactive power balance
def reactive_power_balance(model, i, d):
    return (sum(model.Q[j, i, d] for j, i in model.ol if (j, i) in model.ol) - 
            sum(model.Q[i, j, d] + model.X[i, j] * model.Iqdr[i, j, d] for i, j in model.ol if (i, j) in model.ol) +
            model.QS[i, d] == model.QD[i, d])

model.reactivePowerBalance = Constraint(model.ob, model.od, rule=reactive_power_balance)

# Voltage magnitude drop
def voltage_magnitude_drop(model, i, j, d):
    return (model.Vqdr[i, d] - 2 * (model.R[i, j] * model.P[i, j, d] + model.X[i, j] * model.Q[i, j, d]) -
            model.Z2[i, j] * model.Iqdr[i, j, d] - model.Vqdr[j, d] == 0)

model.voltageMagnitudeDrop = Constraint(model.ol, model.od, rule=voltage_magnitude_drop)

# Current magnitude calculation
def current_magnitude_calculation(model, i, j, d):
    return model.Vqdr[j, d] * model.Iqdr[i, j, d] >= model.P[i, j, d]**2 + model.Q[i, j, d]**2

model.currentMagnitudeCalculation = Constraint(model.ol, model.od, rule=current_magnitude_calculation)

# Current magnitude limit
def current_magnitude_limit(model, i, j, d):
    return model.Iqdr[i, j, d] <= model.Imax[i, j]

model.currentMagnitudeLimit = Constraint(model.ol, model.od, rule=current_magnitude_limit)

# Voltage magnitude limit
def voltage_magnitude_limit(model, i, d):
    return model.Vqdr[i, d] >= 0

model.voltageMagnitudeLimit = Constraint(model.ob, model.od, rule=voltage_magnitude_limit)
