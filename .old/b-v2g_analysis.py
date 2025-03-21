import json
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size':10})

# Caminho do arquivo JSON
file_path = "Pareto/3-evaluations.json"

# Carregar os dados do JSON
with open(file_path, "r") as file:
    data = json.load(file)

# Extração e conversão dos dados
x_values = list(map(float, data.keys()))
y_values_left = [entry["Total energy by V2G"] for entry in data.values()]

# Configuração da figura e do eixo
fig, ax1 = plt.subplots(figsize=(10.5, 7.5))  # 


# Personalização do eixo y (esquerda)
ax1.set_xlabel("V2G usage cost [USD/kWh]")
ax1.set_ylabel("Average energy by V2G [kWh]", color="tab:blue")
ax1.plot(x_values, y_values_left, marker="o", linestyle="-", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Adicionar um marcador específico (triângulo) em um ponto de destaque
highlight_x, highlight_y = 0.09, 78.7872988053755
ax1.plot(highlight_x, highlight_y, marker="s", markersize=10, color="#ff8a84")

# Configurações finais
plt.title("Pareto Front for V2G Analysis")
fig.tight_layout()
plt.grid(True)

# Salvar o gráfico
plt.savefig("Pareto/pareto_front.pdf")
