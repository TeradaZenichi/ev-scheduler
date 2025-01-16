import json
from datetime import datetime, timedelta
import numpy as np

def parse_time(t):
    return datetime.strptime(t, '%H:%M')

def format_time(dt):
    return dt.strftime('%H:%M')

def interpolate_values(times, values, new_times):
    # Convert datetime objects to seconds since start of the day
    time_seconds = [(t - times[0]).total_seconds() for t in times]
    new_time_seconds = [(t - times[0]).total_seconds() for t in new_times]

    interpolated_values = np.interp(new_time_seconds, time_seconds, values)
    return interpolated_values

def interpolate_data(data):
    new_data = {}
    start_time = parse_time('00:00')
    end_time = parse_time('23:55')
    delta = timedelta(minutes=5)

    new_times = []
    current_time = start_time
    while current_time <= end_time:
        new_times.append(current_time)
        current_time += delta

    times = [parse_time(t) for t in data.keys()]
    values = [v for v in data.values()]
    new_values = interpolate_values(times, values, new_times)
    new_data = {format_time(t): float('{:.3f}'.format(v)) for t, v in zip(new_times, new_values)}

    return new_data

with open('data/scenarios.json') as f:
    scenarios = json.load(f)

for scenario_key in scenarios.keys():
    scenario = scenarios[scenario_key]
    for data_key in ['pv', 'load']:  # Ajuste aqui conforme a estrutura dos seus dados
        if data_key in scenario:
            scenario[data_key] = interpolate_data(scenario[data_key])

# Salve ou use `scenarios` conforme necessário


with open('data/scenarios_interpolated.json', 'w') as f:
    json.dump(scenarios, f, indent=4)