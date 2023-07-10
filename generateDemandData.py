# Day - Station - Demand (int Wert für Summe in+out checks)
import csv
import os

data = {}
hourly = True

directory = os.fsencode("data/prepared")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        print(filename)
        open_file = open("data/prepared/" + filename, "r")
        reader = csv.reader(open_file)
        next(reader)
        for row in reader:
            if hourly:
                date = row[3].split(" ")[0] + "-" + row[3].split(" ")[1].split(":")[0]
            else:
                date = row[3].split(" ")[0]
            station = row[2]
            if date in data:
                if station in data[date]:
                    data[date][station] += 1
                else:
                    data[date][station] = 1
            else:
                data[date] = {}
                data[date][station] = 1
            station = row[5]
            if len(station.split()) > 1:
                station = row[6]
            if date in data:
                if station in data[date]:
                    data[date][station] += 1
                else:
                    data[date][station] = 1
            else:
                data[date] = {}
                data[date][station] = 1

if hourly:
    file_name = "station_demand_per_day_hourly.csv"
else:
    file_name = "station_demand_per_day.csv"

with open(file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Station', 'Day', 'Demand'])
    for day, day_data in data.items():
        for station, count in day_data.items():
            writer.writerow([station, day, count])
