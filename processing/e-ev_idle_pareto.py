import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager
import matplotlib.ticker as ticker

# Verificar se a fonte está disponível
font_path = 'Gulliver.otf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    print("Fonte 'Gulliver.otf' não encontrada, usando Times New Roman.")
    plt.rcParams['font.family'] = 'Times New Roman'

# Configurar tamanho de fonte para artigo científico
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# Definir tamanho do gráfico
fig_width_mm = 190 * 2
fig_height_mm = 30 * 2
fig_width_inch = fig_width_mm / 25.4
fig_height_inch = fig_height_mm / 25.4

# Carregar JSON
file_path = "Pareto/5-ev_idle_pareto.json"
with open(file_path, "r") as file:
    data_ev_idle = json.load(file)

df_ev_idle = pd.DataFrame.from_dict(data_ev_idle, orient="index")
df_ev_idle.index = df_ev_idle.index.astype(float)
df_ev_idle.sort_index(inplace=True)

# Ajustar dados escalonados
df_ev_idle["EVidle (Scaled)"] = df_ev_idle["EVidle"] * 0.25

# Função para posicionar legendas sem sobreposição
legend_kwargs1 = {"loc": "upper center", "bbox_to_anchor": (0.3, 1.20), "ncol": 2, "frameon": True}
legend_kwargs2 = {"loc": "upper center", "bbox_to_anchor": (0.7, 1.20), "ncol": 2, "frameon": True}

# Gráfico 1: Objective vs. Minimum allowed EV idle time
fig, ax1 = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
ax1.set_xlabel("Minimum Allowed EV Idle Time [min]")
ax1.set_ylabel("Total Cost [MUSD]", color="tab:blue")
ax1.plot(df_ev_idle.index, df_ev_idle["Objective"]/1e6, marker='o', linestyle='-', label="Total Cost [MUSD]", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2 = ax1.twinx()
ax2.set_ylabel("Annual GHG Emissions [kgCO2-eq]", color="tab:red")
ax2.plot(df_ev_idle.index, df_ev_idle["GHG"], marker='s', linestyle='--', label="GHG Emissions", color="tab:red")
ax2.tick_params(axis='y', labelcolor="tab:red")

ax1.legend(**legend_kwargs1)
ax2.legend(**legend_kwargs2)

plt.savefig("Pareto/e-ev_idle_pareto_1.pdf", dpi=300, bbox_inches="tight")

# Gráfico 2: Múltiplas métricas vs. Minimum allowed EV idle time
fig, ax1 = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
ax1.set_xlabel("Minimum Allowed EV Idle Time [min]")
ax1.set_ylabel("PV and TG Installed Capacity [kW]", color="tab:blue")
line1, = ax1.plot(df_ev_idle.index, df_ev_idle["PPVmax"], marker='o', linestyle='-', label="PV Installed [kW]", color="tab:blue")
line2, = ax1.plot(df_ev_idle.index, df_ev_idle["TG_MAX_CAP"], marker='s', linestyle='--', label="TG Installed [kW]", color="tab:cyan")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2 = ax1.twinx()
ax2.set_ylabel("BESS, V2G Energy, and EDS Usage [MWh]", color="tab:red")
line3, = ax2.plot(df_ev_idle.index, df_ev_idle["EmaxBESS"], marker='^', linestyle='-', label="BESS Installed [kWh]", color="tab:red")
line4, = ax2.plot(df_ev_idle.index, df_ev_idle["Total energy by V2G"] * 0.25, marker='x', linestyle='--', label="Total Energy by V2G [kWh]", color="tab:orange")
ax2.tick_params(axis='y', labelcolor="tab:red")

ax1.legend(**legend_kwargs1)
ax2.legend(**legend_kwargs2)

plt.savefig("Pareto/e-ev_idle_pareto_2.pdf", dpi=300, bbox_inches="tight")

# Gráfico 3: NEVCS vs. Minimum allowed EV idle time
fig, ax1 = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
ax1.set_xlabel("Minimum Allowed EV Idle Time [min]")
ax1.set_ylabel("NEVCS", color="tab:blue")
ax1.plot(df_ev_idle.index, df_ev_idle["NEVCS"], marker='o', linestyle='-', label="NEVCS", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2 = ax1.twinx()
ax2.set_ylabel("EV Idle Time [min]", color="tab:red")
ax2.plot(df_ev_idle.index, df_ev_idle["EVidle (Scaled)"], marker='s', linestyle='--', label="Total EV Idle Time [min]", color="tab:red")
ax2.tick_params(axis='y', labelcolor="tab:red")

ax1.legend(**legend_kwargs1)
ax2.legend(**legend_kwargs2)

plt.savefig("Pareto/e-ev_idle_pareto_3.pdf", dpi=300, bbox_inches="tight")
