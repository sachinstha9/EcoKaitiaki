import json

with open("assets/data.json", "r", encoding='utf-8') as file:
    data = json.load(file)

arr = []

print(data[0].keys())