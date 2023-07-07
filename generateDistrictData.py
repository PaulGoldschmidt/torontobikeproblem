# Day - District - Demand (int Wert für Summe in+out checks)
import csv
import os

# Station ID to District ID
stations = {}
open_file = open("koordinaten.csv", "r")
reader = csv.reader(open_file)
next(reader)
for row in reader:
    open_file_districts = open("zipcoords.csv", "r")
    reader_districts = csv.reader(open_file_districts)
    nearest = 1e9
    nearest_id = ""
    for district in reader_districts:
        distance = abs(
            (float(row[2]) - float(district[1])) * (float(row[3]) - float(district[2])) - (
                        float(row[2]) - float(district[1])) * (
                    float(row[3]) - float(district[2])))
        if distance < nearest:
            nearest = distance
            nearest_id = district[0]
    stations[row[0]] = nearest_id

data = {}
open_file = open("station_demand_per_day.csv", "r")
reader = csv.reader(open_file)
next(reader)
for row in reader:
    if row[0] in stations:
        district = stations[row[0]]
        date = row[1]
        demand = int(row[2])
        if date in data:
            if district in data[date]:
                data[date][district] += demand
            else:
                data[date][district] = demand
        else:
            data[date] = {}
            data[date][district] = demand

with open('district_demand_per_day.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['District', 'Day', 'Demand'])
    for day, day_data in data.items():
        for district, count in day_data.items():
            writer.writerow([district, day, count])
