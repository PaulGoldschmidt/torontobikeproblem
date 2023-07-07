import requests
from bs4 import BeautifulSoup

# Function to scrape bike share stations
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

        # Iterate over the station rows and extract capacity information
        for row in station_rows:
            # Extract station name
            station_name = row.find('td', class_='station-name').text.strip()

            # Extract station capacity
            station_capacity = row.find('td', class_='station-capacity').text.strip()

            # Print station name and capacity
            print(f"Station: {station_name}")
            print(f"Capacity: {station_capacity}")
            print('----------------------')

    else:
        print('Failed to retrieve data from OpenStreetMap.')


# Call the function to scrape bike share stations
scrape_bike_share_stations()
