import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON file
file_path = "Results_pareto/contingency_results.json"  # Path on your computer
with open(file_path, "r") as file:
    data = json.load(file)

# Extract keys (number of contingencies) and "Total Time [s]" values
x = np.array([int(key) for key in data.keys()])  # Convert keys to integers
y = np.array([data[key]["Total Time [s]"] for key in data.keys()])  # Extract total time

# Fit a polynomial model (order 2 or 3 to evaluate complexity)
degree = 2  # Change to 1 (linear), 3 (cubic), etc., to test different models
coeffs = np.polyfit(x, y, degree)  # Polynomial fitting
poly_model = np.poly1d(coeffs)  # Create polynomial function

# Generate predicted values for the trend line
x_fit = np.linspace(min(x), max(x), 100)  # Smooth values for plotting
y_fit = poly_model(x_fit)  # Apply polynomial model

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b', alpha=0.7, label="Real Data")
plt.plot(x_fit, y_fit, linestyle="--", color="r", label=f"Polynomial Fit (Degree {degree})")

# Plot settings
plt.xlabel("Number of Contingencies")
plt.ylabel("Total Time [s]")
plt.title("Relationship Between Contingencies and Total Time")
plt.xticks(range(0, max(x) + 1, max(1, len(x) // 20)), rotation=90)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Display the plot
plt.show()

# Print the fitted equation
print(f"Fitted Equation: {poly_model}")
