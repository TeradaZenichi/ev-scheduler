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
fig_height_mm = 60 * 2
fig_width_inch = fig_width_mm / 25.4
fig_height_inch = fig_height_mm / 25.4

# Carregar JSON
file_path = "Pareto/4-ghg_pareto.json"
with open(file_path, "r") as file:
    data = json.load(file)

df = pd.DataFrame.from_dict(data, orient="index")
df.index = df.index.astype(float)
df.sort_index(inplace=True)

# Ajustar dados escalonados
df["Total energy by V2G (Scaled)"] = df["Total energy by V2G"] * 0.25
df["EDS usage (Scaled)"] = df["EDS usage"] / 1000

# Gráfico 1: Objetivo vs. GHG
plt.figure(figsize=(fig_width_inch, fig_height_inch))
plt.plot(df["GHG"], df["Objective"]/1e6, marker='o', linestyle='-', label="Total Cost")
plt.xlabel("Anual GHG Emissions (kgCO2eq)")
plt.ylabel("Total Cost [MUSD]")
plt.grid(True)
plt.tight_layout()

# Selecionar pontos de interesse
highlight_indices = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
highlight_points = df.iloc[highlight_indices]
plt.scatter(highlight_points["GHG"], highlight_points["Objective"]/1e6, color='red', label="Highlighted Points", zorder=3)

# Imprimir chaves dos pontos destacados
highlight_keys = df.index[highlight_indices]
print("Chaves dos pontos destacados:")
for key in highlight_keys:
    print(key)

# Ajustar legenda para ficar acima do gráfico
legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True)
legend.get_frame().set_alpha(1)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.5)

plt.savefig("Pareto/d-ghg_pareto_1.pdf", dpi=300, bbox_inches="tight")
plt.savefig("Pareto/d-ghg_pareto_1.svg", dpi=300, bbox_inches="tight")

# Gráfico 2: Múltiplas métricas vs. GHG
fig, ax1 = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
plt.grid(True)

ax1.set_xlabel("Yearly GHG Emissions (kgCO2eq)")
ax1.set_ylabel("Power", color="tab:blue")

line1, = ax1.plot(df["GHG"], df["PPVmax"], marker='o', linestyle='-', label="PV installed [kW]", color="tab:blue")
line2, = ax1.plot(df["GHG"], df["TG_MAX_CAP"], marker='s', linestyle='--', label="TG installed [kW]", color="tab:cyan")

ax1.tick_params(axis='y', labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Energy", color="tab:red")

line3, = ax2.plot(df["GHG"], df["EmaxBESS"], marker='^', linestyle='-', label="BESS installed [kWh]", color="tab:red")
line4, = ax2.plot(df["GHG"], df["Total energy by V2G (Scaled)"], marker='x', linestyle='--', label="Total energy by V2G [kWh]", color="tab:orange")
line5, = ax2.plot(df["GHG"], df["EDS usage (Scaled)"], marker='d', linestyle=':', label="EDS usage [MWh]", color="tab:purple")

ax2.tick_params(axis='y', labelcolor="tab:red")

# Ajustar legendas para ficarem acima do gráfico
legend1 = ax1.legend(handles=[line1, line2], loc="upper center", bbox_to_anchor=(0.3, 1.2), ncol=2, frameon=True)
legend1.get_frame().set_alpha(1)
legend1.get_frame().set_facecolor("white")
legend1.get_frame().set_edgecolor("black")
legend1.get_frame().set_linewidth(1.5)

legend2 = ax2.legend(handles=[line3, line4, line5], loc="upper center", bbox_to_anchor=(0.7, 1.2), ncol=2, frameon=True)
legend2.get_frame().set_alpha(1)
legend2.get_frame().set_facecolor("white")
legend2.get_frame().set_edgecolor("black")
legend2.get_frame().set_linewidth(1.5)

plt.savefig("Pareto/d-ghg_pareto_2.pdf", dpi=300, bbox_inches="tight")
