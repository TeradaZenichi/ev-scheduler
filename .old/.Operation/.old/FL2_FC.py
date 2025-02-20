from pyomo.environ import *
from time import time
t1 = time()

m = AbstractModel()

# declaração de conjuntos e parâmetros
m.ob = Set()                     # conjunto de nós
m.ol = Set(dimen=2)              # conjunto de circuitos
m.od = Set()                     # conjunto de níveis de demanda

m.obout = Set(m.ob, within=m.ob, initialize=[])
m.obin  = Set(m.ob, within=m.ob, initialize=[])

def Populate_In_and_Out(m):
    for i, j in m.ol:
        m.obin[j].add(i)
        m.obout[i].add(j)
m.In_n_Out = BuildAction(rule=Populate_In_and_Out)


m.Tb    = Param(m.ob)                       # tipo de barra 0: carga, 1: SE
m.PD    = Param(m.ob, m.od)                 # potência ativa de demanda no nó i
m.QD    = Param(m.ob, m.od)                 # potência reativa de demanda no nó i
m.R     = Param(m.ol, mutable=True)         # resistência no circuito ij
m.X     = Param(m.ol, mutable=True)         # reatância no circuito ij
m.Z2    = Param(m.ol, mutable=True)         # impedância no circuito ij ao quadrado
m.Imax  = Param(m.ol)                       # limite máximo da magnitude de corrente no circuito ij
m.alpha = Param(m.od)                       # número de horas no nível d
m.cls   = Param(m.od)                       # custo das perdas de energia no nível d
m.Vnom  = Param(initialize=11.0 / sqrt(3),  # magnitude da tensão nominal
               mutable=True)
m.Y     = Param(initialize=50)              # número de linearizações
m.ms    = Param(m.ol, RangeSet(0,m.Y),      # inclinação do y-esimo bloco do fluxo no circuito ij
                mutable=True, initialize=0)               
m.DeltS = Param(m.ol, mutable=True)         # limite superior de cada bloco de linearizações

# declaração das variáveis
m.Vqdr  = Var(m.ob, m.od, domain=Reals, initialize=0.0)   # variável que representa o quadrado de V[i,d]
m.PS    = Var(m.ob, m.od, domain=Reals)                   # potência ativa fornecida pela subestação no nó i 
m.QS    = Var(m.ob, m.od, domain=Reals)                   # potência reativa fornecida pela subestação no nó i
m.Iqdr  = Var(m.ol, m.od, initialize=0)                   # variável que representa o quadrado de I[i,j,d] 
m.P     = Var(m.ol, m.od, domain=Reals, initialize=0.0)   # fluxo de potência ativa no circuito ij
m.Q     = Var(m.ol, m.od, domain=Reals, initialize=0.0)   # fluxo de potência reativa no circuito ij
m.DeltP = Var(m.ol, m.od, RangeSet(0,m.Y))                # valor do y-ésimo bloco de P[i,j,d]
m.DeltQ = Var(m.ol, m.od, RangeSet(0,m.Y))                # valor do y-ésimo bloco de Q[i,j,d]
m.PM    = Var(m.ol, m.od, within=NonNegativeReals)        # variáveis auxiliares não negativas, modelar |P[i,j,d]|   
m.Pm    = Var(m.ol, m.od, within=NonNegativeReals)        # variáveis auxiliares não negativas, modelar |P[i,j,d]| 
m.QM    = Var(m.ol, m.od, within=NonNegativeReals)        # variáveis auxiliares não negativas, modelar |Q[i,j,d]|  
m.Qm    = Var(m.ol, m.od, within=NonNegativeReals)        # variáveis auxiliares não negativas, modelar |Q[i,j,d]| 

# definição da função objetivo
def custo(m):
    return sum(m.cls[d]*m.alpha[d]*sum(m.R[i,j] * m.Iqdr[i,j,d] for i,j in m.ol) for d in m.od) 
m.custo = Objective(rule=custo, sense=minimize)

def balance_potencia_activa(m, i, d):
    return sum(m.P[j,i,d] for j in m.obin[i]) - \
           sum(m.P[i,j,d] + m.R[i,j] * m.Iqdr[i,j,d] for j in m.obout[i]) + m.PS[i,d] == m.PD[i,d]
m.balance_potencia_activa = Constraint(m.ob, m.od, rule=balance_potencia_activa)

def balance_potencia_reativa(m, i, d):
    return sum(m.Q[j,i,d] for j in m.obin[i]) - \
        sum(m.Q[i,j,d] + m.X[i,j] * m.Iqdr[i,j,d] for j in m.obout[i]) + m.QS[i,d] == m.QD[i,d]
m.balance_potencia_reativa = Constraint(m.ob, m.od, rule=balance_potencia_reativa)

def queda_magnitude_tensao(m, i, j, d):
    return m.Vqdr[i,d] - 2*(m.R[i,j]*m.P[i,j,d]+m.X[i,j]*m.Q[i,j,d]) - m.Z2[i,j]*m.Iqdr[i,j,d] - m.Vqdr[j,d] == 0
m.queda_magnitude_tensao = Constraint(m.ol, m.od, rule=queda_magnitude_tensao)

def calculo_magnitude_corrente(m, i, j, d):
    return m.Vnom**2 * m.Iqdr[i,j,d] == sum(m.ms[i,j,y] * m.DeltP[i,j,d,y] for y in RangeSet(0, m.Y)) + \
                                        sum(m.ms[i,j,y] * m.DeltQ[i,j,d,y] for y in RangeSet(0, m.Y))
m.calculo_magnitude_corrente = Constraint(m.ol, m.od, rule=calculo_magnitude_corrente)

def limite_magnitude_corrente(m, i, j, d):
    return 0 <= m.Iqdr[i,j,d]
m.limite_magnitude_corrente = Constraint(m.ol, m.od, rule=limite_magnitude_corrente)

def limite_magnitude_tensão(m, i, d):
    return 0 <= m.Vqdr[i,d]
m.limite_magnitude_tensão = Constraint(m.ob, m.od, rule=limite_magnitude_tensão)

def calculo_valor_absoluto_P(m, i, j, d):
    return m.PM[i, j, d] - m.Pm[i, j, d] == m.P[i, j, d]
m. calculo_valor_absoluto_P = Constraint(m.ol, m.od, rule=calculo_valor_absoluto_P)

def calculo_DeltP(m, i, j, d):
    return m.PM[i, j, d] + m.Pm[i, j, d] == sum(m.DeltP[i, j, d, y] for y in RangeSet(0, m.Y))
m. calculo_DeltP = Constraint(m.ol, m.od, rule=calculo_DeltP)

def calculo_valor_absoluto_Q(m, i, j, d):
    return m.QM[i, j, d] - m.Qm[i, j, d] == m.Q[i, j, d]
m. calculo_valor_absoluto_Q = Constraint(m.ol, m.od, rule=calculo_valor_absoluto_Q)

def calculo_DeltQ(m, i, j, d):
    return m.QM[i, j, d] + m.Qm[i, j, d] == sum(m.DeltQ[i, j, d, y] for y in RangeSet(0, m.Y))
m. calculo_DeltQ = Constraint(m.ol, m.od, rule=calculo_DeltQ)

def limite_DeltP(m, i, j, d, y):
    return inequality(0, m.DeltP[i, j, d, y], m.DeltS[i, j])
m. limite_DeltP = Constraint(m.ol, m.od, RangeSet(0, m.Y), rule=limite_DeltP)

def limite_DeltQ(m, i, j, d, y):
    return inequality(0, m.DeltQ[i, j, d, y], m.DeltS[i, j])
m. limite_DeltQ = Constraint(m.ol, m.od, RangeSet(0, m.Y), rule=limite_DeltQ)

file = "sistema34nos.dat"
instance = m.create_instance(file)

# fixar a magnitude de tensão da subestação no valor nominal
for i in instance.ob:
    for d in instance.od:
        if instance.Tb[i] == 1:
            instance.Vqdr[i, d].fix(instance.Vnom**2)
            
# fixar a injeção de potencia ativa nos nós de carga iguais a zero
for i in instance.ob:
    for d in instance.od:
        if instance.Tb[i] == 0:
            instance.PS[i, d].fix(0)
            
# fixar a injeção de potencia reativa nos nós de carga iguais a zero
for i in instance.ob:
    for d in instance.od:
        if instance.Tb[i] == 0:
            instance.QS[i, d].fix(0)        

        
# nomalização e calculo da impedancia ao quadrado
for i, j in instance.ol:
    instance.R[i, j] = instance.R[i, j] / 1000      # de ohm para kohm
    instance.X[i, j] = instance.X[i, j] / 1000      # de ohm para kohm
    instance.Z2[i, j] = instance.R[i, j]**2 + instance.X[i, j]**2
    
# calculo inclinação do y-esimo bloco e seu limite superior
for i, j in instance.ol:
    instance.DeltS[i, j] = instance.Vnom * instance.Imax[i, j] / instance.Y
    for y in range(0, instance.Y.value):
        instance.ms[i, j, y] = (2 * y - 1) * instance.DeltS[i, j]


solver = SolverFactory('gurobi')
results = solver.solve(instance, tee=True)
print("results.solver.status: " + str(results.solver.status))
print("results.solver.termination_condition: " + str(results.solver.termination_condition))
print('Custo total: %.4f' % instance.custo())

print("results.solver.status: " + str(results.solver.status))
print("results.solver.termination_condition: " + str(results.solver.termination_condition))
print('Custo total: %.4f' % instance.custo())

print("\nRESUMO")
print("Nivel de demanda                      :", end='')
for d in instance.od:
    print(f'{d: >10d}', end='')
print()

print("Perdas de potencia ativa (kW)         :", end='')
for d in instance.od:
    loss = sum(instance.R[i, j].value * instance.Iqdr[i, j, d].value for (i, j) in instance.ol)
    print(f'{loss: >10.2f}', end='')
print()

print("Perdas de potencia reativa (kVAr)     :", end='')
for d in instance.od:
    loss = sum(instance.X[i, j].value * instance.Iqdr[i, j, d].value for (i, j) in instance.ol)
    print(f'{loss: >10.2f}', end='')
print()

print("Magnitude de tensao minima (kV)       :", end='')
for d in instance.od:
    min_V = min(instance.Vqdr[j, d].value for j in instance.ob)
    print(f'{min_V ** 0.5: >10.4f}', end='')
print()

print("Magnitude de tensao minima (pu)       :", end='')
for d in instance.od:
    min_V = min(instance.Vqdr[j, d].value for j in instance.ob)
    print(f'{(min_V ** 0.5) / instance.Vnom.value: >10.4f}', end='')
print()

print("No da magnitude de tensao minima      :", end='')
for d in instance.od:
    min_i = min(i for i in instance.ob if instance.Vqdr[i, d].value == min(instance.Vqdr[j, d].value for j in instance.ob))
    print(f'{min_i: >10d}', end='')
print()

print("Potencia ativa da subestacao (kW)     :", end='')
for d in instance.od:
    min_i = min(i for i in instance.ob if instance.Tb[i] == 1)
    print(f'{instance.PS[min_i, d].value: >10.2f}', end='')
print()

print("Potencia reativa da subestacao (kVAr) :", end='')
for d in instance.od:
    min_i = min(i for i in instance.ob if instance.Tb[i] == 1)
    print(f'{instance.QS[min_i, d].value: >10.2f}', end='')
print()

print("Demanda de potencia ativa (kW)        :", end='')
for d in instance.od:
    demand = sum(instance.PD[i, d] for i in instance.ob)
    print(f'{demand: >10.2f}', end='')
print()

print("Demanda de potencia reativa (kVAr)    :", end='')
for d in instance.od:
    demand = sum(instance.QD[i, d] for i in instance.ob)
    print(f'{demand: >10.2f}', end='')
print()


print()
print()
print("MAGNITUDES DE TENSAO (kV)")
print("    Nivel de demanda\n  No", end='')
for d in instance.od:
    print(f' {d: >8d}', end='')
print()
for i in instance.ob:
    print(f'{i: >4d}', end='')
    for d in instance.od:
        voltage = instance.Vqdr[i, d].value ** 0.5
        print(f' {voltage: >8.4f}', end='')
    print()
    
    
print()
print()
print("MAGNITUDES DE FLUXO DE CORRENTE (A)")
print("         Nivel de demanda\n Circuito", end='')
for d in instance.od:
    print(f' {d: >8d}', end='')
print("     Imax\n")
for i, j in instance.ol:
    print(f'{i: >4d} {j: >4d}', end='')
    for d in instance.od:
        current_flow = instance.Iqdr[i, j, d].value ** 0.5
        print(f' {current_flow: >8.2f}', end='')
    imax_value = instance.Imax[i, j]
    print(f' {imax_value: >8.2f}')
    
    
print()
print()    
print("ERRO ABSOLUTO DA APROXIMACAO DA MAGNITUDE DE FLUXO DE CORRENTE (%)")
print("         Nivel de demanda\n Circuito", end='')
for d in instance.od:
    print(f' {d: >8d}', end='')
print()
for i, j in instance.ol:
    print(f'{i: >4d} {j: >4d}', end='')
    for d in instance.od:
        P = instance.P[i, j, d].value
        Q = instance.Q[i, j, d].value
        Vqdr = instance.Vqdr[j, d].value
        Iqdr = instance.Iqdr[i, j, d].value
        error = 100*abs(((P ** 2 + Q ** 2)/Vqdr)**0.5 - Iqdr**0.5) / ((P**2 + Q**2)/Vqdr)**0.5
        #error = Iqdr
        print(f' {error: >8.2f}', end='')
    print()
    
    
print()
print()
print("FLUXO DE POTENCIA ATIVA (kW)")
print("         Nivel de demanda\n Circuito", end='')
for d in instance.od:
    print(f' {d: >8d}', end='')
print()
for i, j in instance.ol:
    print(f'{i: >4d} {j: >4d}', end='')
    for d in instance.od:
        active_power = instance.P[i, j, d].value
        print(f' {active_power: >8.2f}', end='')
    print() 
    
    
print()  
print()
print("FLUXO DE POTENCIA REATIVA (kVAr)")
print("         Nivel de demanda\n Circuito", end='')
for d in instance.od:
    print(f' {d: >8d}', end='')
print()

# Print reactive power for each (i, j) and d
for i, j in instance.ol:
    print(f'{i: >4d} {j: >4d}', end='')
    for d in instance.od:
        reactive_power = instance.Q[i, j, d].value
        print(f' {reactive_power: >8.2f}', end='')
    print()    
    

print()
print("PERDAS DE POTENCIA ATIVA (kW)")
print("         Nivel de demanda\n Circuito", end='')
for d in instance.od:
    print(f' {d: >8d}', end='')
print()

# Print active power losses for each (i, j) and d
for i, j in instance.ol:
    print(f'{i: >4d} {j: >4d}', end='')
    for d in instance.od:
        R =  instance.R[i, j].value
        Iqdr = instance.Iqdr[i, j, d].value
        active_power_loss = R * Iqdr 
        print(f' {active_power_loss: >8.2f}', end='')
    print()
    

t2 = time()

print()
print()
print(f"Elapsed time  : {(t2 - t1): >10.2f}s")
print()
print()





