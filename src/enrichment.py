import requests
import time

def get_coords_from_postcode(postcode: str) -> dict | None:
    """Convert a UK postcode to lat/long using postcodes.io"""
    url = f"https://api.postcodes.io/postcodes/{postcode.replace(' ', '')}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            result = resp.json()["result"]
            return {"lat": result["latitude"], "lng": result["longitude"], "region": result["region"]}
    except Exception as e:
        print(f"Postcode lookup failed for {postcode}: {e}")
    return None

def get_crime_data(lat: float, lng: float, date: str, retries: int = 2) -> dict:
    url = f"https://data.police.uk/api/crimes-street/all-crime?lat={lat}&lng={lng}&date={date}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                crimes = resp.json()
                vehicle_crimes = sum(1 for c in crimes if c["category"] == "vehicle-crime")
                return {"total_crimes_nearby": len(crimes), "vehicle_crimes_nearby": vehicle_crimes}
        except Exception as e:
            print(f"Crime lookup attempt {attempt+1} failed for {lat},{lng},{date}: {e}")
            time.sleep(1)
    return {"total_crimes_nearby": None, "vehicle_crimes_nearby": None}

def get_weather_data(lat: float, lng: float, date: str) -> dict:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lng}&start_date={date}&end_date={date}"
        f"&daily=precipitation_sum,windspeed_10m_max,temperature_2m_max"
        f"&timezone=Europe/London"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            daily = resp.json()["daily"]
            return {
                "rainfall_mm": daily["precipitation_sum"][0],
                "max_windspeed_kmh": daily["windspeed_10m_max"][0],
                "max_temp_c": daily["temperature_2m_max"][0],
            }
    except Exception as e:
        print(f"Weather lookup failed for {lat},{lng},{date}: {e}")
    return {"rainfall_mm": None, "max_windspeed_kmh": None, "max_temp_c": None}

def enrich_claim(postcode: str, claim_date: str) -> dict:
    coords = get_coords_from_postcode(postcode)
    if not coords:
        return {}

    crime_month = claim_date[:7]
    crime = get_crime_data(coords["lat"], coords["lng"], crime_month)
    weather = get_weather_data(coords["lat"], coords["lng"], claim_date)

    return {
        "postcode": postcode,
        "region": coords["region"],
        **crime,
        **weather,
    }
