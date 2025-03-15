import json
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
import matplotlib

font_path = 'Gulliver.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'Gulliver'
plt.rcParams['font.sans-serif'] = prop.get_name()

plt.rcParams.update({
    "font.size": 10,  # Tamanho de fonte recomendado para artigos científicos (8-12 pt)
    "axes.labelsize": 10,  # Tamanho dos rótulos dos eixos
    "axes.titlesize": 10,  # Tamanho do título
    "xtick.labelsize": 10,  # Tamanho dos ticks do eixo X
    "ytick.labelsize": 10,  # Tamanho dos ticks do eixo Y
    "legend.fontsize": 10,  # Tamanho da legenda
})

# Escolha um tamanho da tabela (em mm)
fig_width_mm = 90 * 2   # Exemplo: 90 mm para Single Column
fig_height_mm = 60 * 2  # Defina a altura manualmente

# Converter mm para polegadas
fig_width_inch = fig_width_mm / 25.4
fig_height_inch = fig_height_mm / 25.4


# Load the JSON data
with open('Pareto/3-evaluations.json', 'r') as f:
    data = json.load(f)

# Initialize lists for prices, total costs, and V2G energy
prices = []
total_costs = []
total_energy_v2g = []

# Extract relevant data from the JSON
for key, values in data.items():
    prices.append(float(key))
    total_costs.append(values["Total Cost"])
    total_energy_v2g.append(values["Total energy by V2G"])

# Calculate effective energy (usage of the V2G service) by multiplying by 0.25
effective_energy = [energy * 0.25 for energy in total_energy_v2g]

# ---------------------------------------------------
# 1. Define Utility Functions for Both Objectives
# ---------------------------------------------------
# Objective 1: Maximize compensation for users (price)
price_min = min(prices)
price_max = max(prices)
# Normalize prices to a range of 0 to 1
utility_compensation = [(p - price_min) / (price_max - price_min) for p in prices]

# Objective 2: Maximize V2G service usage (effective energy)
energy_min = min(effective_energy)
energy_max = max(effective_energy)
# Normalize effective energy to a range of 0 to 1
utility_usage = [(e - energy_min) / (energy_max - energy_min) for e in effective_energy]

# ---------------------------------------------------
# 2. Calculate the Nash Product for Each Solution
# ---------------------------------------------------
# The Nash product is the product of the two normalized utilities
nash_product = [u_comp * u_usage for u_comp, u_usage in zip(utility_compensation, utility_usage)]

# Find the index of the maximum Nash product (the equilibrium)
max_index = np.argmax(nash_product)
equilibrium_price = prices[max_index]
equilibrium_nash = nash_product[max_index]

print("Nash Equilibrium Results:")
print("Equilibrium Price (compensation):", equilibrium_price)
print("Nash Product:", equilibrium_nash)
print("Utility (Compensation):", utility_compensation[max_index])
print("Utility (Usage):", utility_usage[max_index])

# ---------------------------------------------------
# 3. Plotting the Results in Two Separate Figures
# ---------------------------------------------------
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(fig_width_inch, fig_height_inch)


# Plot 1: Effective V2G Usage vs Price
plt.figure(figsize=(fig_width_inch, fig_height_inch))

plt.plot(prices, effective_energy, marker='o', linestyle='-', color='blue', label='Effective V2G Usage')
plt.xlabel('Price (USD)')
plt.ylabel('Effective V2G Usage [kWh]')
plt.title('Effective V2G Usage vs Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Pareto/c-evaluations.pdf")
plt.close("all")


# Plot 2: Normalized Utilities and Nash Product vs Price
plt.figure(figsize=(fig_width_inch, fig_height_inch))


plt.plot(prices, utility_compensation, marker='s', linestyle='--', label='Compensation Utility')
plt.plot(prices, utility_usage, marker='^', linestyle='--', label='Usage Utility')
plt.plot(prices, nash_product, marker='o', linestyle='-', color='red', label='Nash Product')
plt.scatter(equilibrium_price, equilibrium_nash, color='black', s=100,
            label=f'Equilibrium: {equilibrium_price:.2f} USD')
plt.xlabel('Price (USD)')
plt.ylabel('Normalized Value')
plt.title('Normalized Utilities and Nash Product vs Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Pareto/c-nash_product.pdf")
