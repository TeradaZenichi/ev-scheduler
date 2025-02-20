import json
from datetime import datetime, timedelta, time
import json


# Load the JSON file
with open('data/cost.json') as file:
    data = json.load(file)

with open('data/old/scenarios15m.json') as file:
    scenarios = json.load(file)

Ωs = scenarios.keys()
Ωt = data.keys()

new = dict()

for s in Ωs:
    new.update({s: {"pv": {}, "load": {}, "prob": scenarios[s]['prob'], "comment": scenarios[s]['comment']} })
    for n,t in enumerate(Ωt):
        new[s]['pv'].update({t: scenarios[s]['pv'][n]})
        new[s]['load'].update({t: scenarios[s]['load'][n]})


# Save the new dictionary as a JSON file
with open('data/scenarios_new.json', 'w') as file:
    json.dump(new, file)

a = 1