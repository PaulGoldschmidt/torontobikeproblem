
import csv
import requests
from bs4 import BeautifulSoup

# Function to scrape bike share stations and save as CSV
def scrape_bike_share_stations():
    # URL of the Toronto Bike Share OpenStreetMap page
    url = 'https://www.openstreetmap.org/directory/josm?query=Toronto%20Bike%20Share#map=13/43.6700/-79.3900'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all the station rows
        station_rows = soup.find_all('tr', class_='station-row')

        # Create a CSV file and write header row
        with open('bike_share_stations.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Station', 'Capacity', 'Location', 'ID'])

            # Iterate over the station rows and extract information
            for row in station_rows:
                # Extract station name
                station_name = row.find('td', class_='station-name').text.strip()

                # Extract station capacity
                station_capacity = row.find('td', class_='station-capacity').text.strip()

                # Extract station location
                station_location = row.find('td', class_='station-location').text.strip()

                # Extract station ID
                station_id = row.find('td', class_='station-id').text.strip()

                # Write station information to CSV
                writer.writerow([station_name, station_capacity, station_location, station_id])

        print('Data saved successfully.')
    else:
        print('Failed to retrieve data from OpenStreetMap.')


# Call the function to scrape bike share stations and save as CSV
scrape_bike_share_stations()
