import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.interpolate import griddata

# import matplotlib as mpl

# mpl.rc('font', family='serif', serif='cmr10')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size':10})

# Load the JSON data from the file
with open('Pareto/ghg_ev_idle_pareto.json', 'r') as f:
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
# Lower is better, so utility = (max - value) / (max - min)
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
fig = plt.figure(figsize=(12, 6))


ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_xlabel('Avarage EV Idle Time [min]')
ax.set_ylabel('GHG Emission [ton CO2eq]')
ax.set_zlabel('Total Cost [MUSD]')
# ax.set_title('Surface Plot: Objective vs EV Idle Time and GHG')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.07)

# Mark the equilibrium point on the surface
ax.scatter(equilibrium_solution["EVidle"], equilibrium_solution["GHG"], equilibrium_solution["Objective"],
           color='red', s=50, marker='o', label=f'Equilibrium Nash Product')
ax.legend()
# Após criar o plot 3D:
ticks = ax.get_xticks()         # Obtém as posições dos ticks
ax.set_xticklabels(ticks[::-1])   # Define os rótulos na ordem invertida


# Rotate the view before saving (adjust elev and azim as desired)
ax.view_init(elev=30, azim=130)

plt.tight_layout()
# Save the figure to PDF with the rotated view
plt.savefig('Pareto/surface_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()
