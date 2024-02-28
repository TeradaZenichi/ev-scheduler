from datetime import datetime, timedelta
import json

def interpolate_data(data):
    # Converte os horários em objetos datetime para facilitar o cálculo
    times = [datetime.strptime(time, "%H:%M") for time in data.keys()]
    values = list(data.values())

    interpolated_data = {}
    current_time = times[0]
    end_time = times[-1] + timedelta(minutes=15)  # Vai um intervalo além para incluir o último ponto

    while current_time <= end_time:
        previous_times = [time for time in times if time <= current_time]
        next_times = [time for time in times if time >= current_time]

        if not previous_times:  # Se current_time é antes do primeiro horário, use o primeiro valor
            interpolated_value = values[0]
        elif not next_times:  # Se current_time é depois do último horário, use o último valor
            interpolated_value = values[-1]
        else:
            previous_time = max(previous_times)
            next_time = min(next_times)
            previous_index = times.index(previous_time)
            next_index = times.index(next_time)
            previous_value = values[previous_index]
            next_value = values[next_index]

            if previous_time != next_time:  # Evita divisão por zero
                fraction = (current_time - previous_time) / (next_time - previous_time)
                interpolated_value = previous_value + (next_value - previous_value) * fraction
            else:
                interpolated_value = previous_value

        interpolated_data[current_time.strftime("%H:%M")] = interpolated_value
        current_time += timedelta(minutes=5)

    return interpolated_data

with open('data/cost.json') as f:
    data = json.load(f)

# Interpola os dados
interpolated_data = interpolate_data(data)

# Opcional: Converte o dicionário interpolado para JSON
json_data = json.dumps(interpolated_data, indent=4)
print(json_data)

# Opcional: Salva o JSON em um arquivo
with open('data/cost_interpolated.json', 'w') as f:
    f.write(json_data)