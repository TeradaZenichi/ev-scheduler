import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.interpolate import griddata
import os
from matplotlib import font_manager

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
fig_width_mm = 90 * 2
fig_height_mm = 90 * 2
fig_width_inch = fig_width_mm / 25.4
fig_height_inch = fig_height_mm / 25.4

# Load the JSON data from the file
with open('Pareto/6-ghg_ev_idle_pareto.json', 'r') as f:
    data = json.load(f)

# Initialize lists for each point:
# x-axis: EVidle, y-axis: GHG, z-axis: Objective (Total Cost)
ev_idle = []
ghg = []
objective = []
keys_list = []

for key, values in data.items():
    keys_list.append(key)
    ev_idle.append(values["EVidle"]/(30 * 9))
    ghg.append(values["GHG"]/1000)
    objective.append(values["Objective"]/1e6)

# Compute normalized utilities for minimization objectives.
def compute_utility(values):
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        return [1.0 for _ in values]
    return [(v_max - v) / (v_max - v_min) for v in values]

utility_ev = compute_utility(ev_idle)
utility_ghg = compute_utility(ghg)
utility_obj = compute_utility(objective)

# Compute Nash product (product of the three utilities)
nash_product = [ue * ug * uo for ue, ug, uo in zip(utility_ev, utility_ghg, utility_obj)]

# Identify the Nash equilibrium (point with the highest Nash product)
max_index = np.argmax(nash_product)
equilibrium_key = keys_list[max_index]
equilibrium_solution = {
    "EVidle": ev_idle[max_index],
    "GHG": ghg[max_index],
    "Objective": objective[max_index],
    "Utility EVidle": utility_ev[max_index],
    "Utility GHG": utility_ghg[max_index],
    "Utility Objective": utility_obj[max_index],
    "Nash Product": nash_product[max_index]
}

print("Nash Equilibrium Solution:")
print("Key (Tuple):", equilibrium_key)
print("EV Idle:", equilibrium_solution["EVidle"])
print("GHG:", equilibrium_solution["GHG"])
print("Objective (Cost):", equilibrium_solution["Objective"])
print("Utility EVidle:", equilibrium_solution["Utility EVidle"])
print("Utility GHG:", equilibrium_solution["Utility GHG"])
print("Utility Objective:", equilibrium_solution["Utility Objective"])
print("Nash Product:", equilibrium_solution["Nash Product"])

# Create a grid for interpolation of the Objective values over the (EVidle, GHG) space
yi = np.linspace(min(ev_idle), max(ev_idle), 100)
xi = np.linspace(min(ghg), max(ghg), 100)
xi, yi = np.meshgrid(xi, yi)
points = np.column_stack((ev_idle, ghg))
zi = griddata(points, objective, (xi, yi), method='linear')

# Create the 3D surface plot
fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_xlabel('Average EV Idle Time (minutes)')
ax.set_ylabel('GHG Emissions (tons CO2-eq)')
ax.set_zlabel('Total Cost (MUSD)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.10)
ticks = ax.get_xticks()         # Obtém as posições dos ticks
ax.set_xticklabels(ticks[::-1]) # Inverte os valores dos ticks
# Mark the equilibrium point on the surface
ax.scatter(equilibrium_solution["EVidle"], equilibrium_solution["GHG"], equilibrium_solution["Objective"],
           color='red', s=50, marker='o',  depthshade=False, label='Nash Equilibrium Point')
ax.legend()

# Rotate the view before saving (adjust elev and azim as desired)
ax.view_init(elev=30, azim=130)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# Save the figure to PDF with the rotated view
plt.savefig('Pareto/f-surface_plot.pdf', format='pdf', bbox_inches='tight')
# plt.show()