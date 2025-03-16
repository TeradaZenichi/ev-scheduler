import json
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

# Verificar se a fonte está disponível
font_path = 'Gulliver.otf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

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
fig_width_mm = 90 * 2
fig_height_mm = 40 * 2
fig_width_inch = fig_width_mm / 25.4
fig_height_inch = fig_height_mm / 25.4

# Load JSON data for scenarios
scenarios_file_path = "data/scenarios.json"
with open(scenarios_file_path, "r") as file:
    scenarios_data = json.load(file)

# Extract timestamps
timestamps = list(scenarios_data["1"]["pv"].keys())

# Extract PV and Load data for each scenario
scenario_pv = {}
scenario_load = {}

for scenario, values in scenarios_data.items():
    pv_values = list(values["pv"].values())
    load_values = list(values["load"].values())

    scenario_pv[scenario] = sum(pv_values) / len(pv_values)  # Average PV generation
    scenario_load[scenario] = sum(load_values) / len(load_values)  # Average Load

# Identify scenarios corresponding to low, medium, and high PV and Load
sorted_pv_scenarios = sorted(scenario_pv, key=scenario_pv.get)
sorted_load_scenarios = sorted(scenario_load, key=scenario_load.get)

low_pv_scenario = sorted_pv_scenarios[0]
medium_pv_scenario = sorted_pv_scenarios[len(sorted_pv_scenarios) // 2]
high_pv_scenario = sorted_pv_scenarios[-1]

low_load_scenario = sorted_load_scenarios[0]
medium_load_scenario = sorted_load_scenarios[len(sorted_load_scenarios) // 2]
high_load_scenario = sorted_load_scenarios[-1]

# Extract time-series data for each selected scenario
low_pv_values = list(scenarios_data[low_pv_scenario]["pv"].values())
medium_pv_values = list(scenarios_data[medium_pv_scenario]["pv"].values())
high_pv_values = list(scenarios_data[high_pv_scenario]["pv"].values())

low_load_values = list(scenarios_data[low_load_scenario]["load"].values())
medium_load_values = list(scenarios_data[medium_load_scenario]["load"].values())
high_load_values = list(scenarios_data[high_load_scenario]["load"].values())

# Normalize values (max normalization)
max_pv = max(high_pv_values)
max_load = max(high_load_values)

low_pv_values = [v / max_pv for v in low_pv_values]
medium_pv_values = [v / max_pv for v in medium_pv_values]
high_pv_values = [v / max_pv for v in high_pv_values]

low_load_values = [v / max_load for v in low_load_values]
medium_load_values = [v / max_load for v in medium_load_values]
high_load_values = [v / max_load for v in high_load_values]

# Define indices for reduced X-axis ticks (e.g., every hour)
tick_indices = np.linspace(0, len(timestamps) - 1, num=12, dtype=int)
tick_labels = [timestamps[i] for i in tick_indices]

# Plot PV and Load curves with different line styles
plt.figure(figsize=(fig_width_inch, fig_height_inch))

# PV - Continuous lines
plt.plot(timestamps, low_pv_values, linestyle='-', label="PV Low", color="#377eb8", linewidth=2.25)
plt.plot(timestamps, medium_pv_values, linestyle='-', label="PV Medium", color="#ff7f00", linewidth=2.25)
plt.plot(timestamps, high_pv_values, linestyle='-', label="PV High", color="#4daf4a", linewidth=2.25)

# Load - Dashed lines
plt.plot(timestamps, low_load_values, linestyle='--', label="Load Low", color="#984ea3", linewidth=2.25)
plt.plot(timestamps, medium_load_values, linestyle='--', label="Load Medium", color="#ffff33", linewidth=2.25)
plt.plot(timestamps, high_load_values, linestyle='--', label="Load High", color="#e41a1c", linewidth=2.25)

plt.xlabel("Timestamp")
plt.ylabel("Normalized Value")

# Apply reduced ticks
plt.xticks(tick_indices, tick_labels, rotation=45)

plt.legend()
plt.grid(True)

# Save the figure to PDF
plt.savefig("Pareto/g-scenarios.pdf", format='pdf', bbox_inches="tight", dpi=300)
plt.show()
