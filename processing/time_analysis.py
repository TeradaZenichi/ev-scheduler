import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib as mpl
mpl.rc('font', family='serif', serif='cmr10')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size':14})

# Load the JSON file
file_path = "Results_pareto/contingency_results.json"  # Path on your computer
with open(file_path, "r") as file:
    data = json.load(file)

# Extract keys (number of contingencies) and "Total Time [s]" values
x = np.array([int(key) for key in data.keys()])  # Convert keys to integers
y = np.array([data[key]["Total Time [s]"] for key in data.keys()])  # Extract total time
# first 10 values
x = x[:52]
y = y[:52]


# Fit a polynomial model (order 2 or 3 to evaluate complexity)
degree = 1  # Change to 1 (linear), 3 (cubic), etc., to test different models
coeffs = np.polyfit(x, y, degree)  # Polynomial fitting
poly_model = np.poly1d(coeffs)  # Create polynomial function

# Generate predicted values for the trend line
x_fit = np.linspace(min(x), max(x), 100)  # Smooth values for plotting
y_fit = poly_model(x_fit)  # Apply polynomial model

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(x, y, marker='o', linestyle='-', color='g', alpha=0.7, label="Number of Contingencies")
plt.plot(x_fit, y_fit, linestyle="--", color="black", label=f"Polynomial Fit - Number of Contingencies (Degree {degree})")

# Plot settings
plt.xlabel("Number of Parameters")
plt.ylabel("Total Time [s]")
plt.title("Relationship between number of parameters and total time")
plt.xticks(range(0, max(x) + 1, max(1, len(x) // 20)), rotation=90)
plt.grid(True, linestyle="--", alpha=0.6)

file_path = "Results_pareto/ev_number_results.json"  
with open(file_path, "r") as file:
    data = json.load(file)

# Extract keys (number of total electric vehicles) and "Total Time [s]" values
x = np.array([int(key) for key in data.keys()])  # Convert keys to integers
y = np.array([data[key]["Total Time [s]"] for key in data.keys()])  # Extract total time
x = x[:52]
y = y[:52]

# Fit a polynomial model (order 2 or 3 to evaluate complexity)
degree = 1  # Change to 1 (linear), 3 (cubic), etc., to test different models
coeffs = np.polyfit(x, y, degree)  # Polynomial fitting
poly_model = np.poly1d(coeffs)  # Create
# polynomial
# function
x_fit = np.linspace(min(x), max(x), 100)  # Smooth values for plotting
y_fit = poly_model(x_fit)  # Apply polynomial

plt.plot(x, y, marker='o', linestyle='-', color='b', alpha=0.7, label="Number of EVs")
plt.plot(x_fit, y_fit, linestyle="--", color="r", label=f"Polynomial Fit - Number of EVs (Degree {degree})")
plt.tight_layout()
plt.legend()

plt.savefig("Results_pareto/contingency_ev_time.pdf")

# Print the fitted equation
print(f"Fitted Equation: {poly_model}")