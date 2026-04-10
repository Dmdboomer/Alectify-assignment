"""
Generate synthetic GPS + sleep data for truck drivers over ~1 year.

Each driver has a home base and drives routes. When the truck is stationary
for 4+ hours (especially overnight), we label that as a sleep event.
The data captures realistic patterns:
  - Drivers tend to sleep at consistent times with mild drift
  - Weekly patterns (less driving on weekends for some)
  - Seasonal variation (longer rest in winter, shorter in summer)
  - Individual driver habits (early sleeper vs late sleeper)
"""

import csv
import random
import os
from datetime import datetime, timedelta
import math

random.seed(42)

# --- Configuration ---
NUM_DRIVERS = 10
START_DATE = datetime(2025, 4, 1)
END_DATE = datetime(2026, 4, 1)  # ~1 year
OUTPUT_DIR = "data"

# US cities as route waypoints (lat, lon)
CITIES = {
    "Dallas_TX":        (32.7767, -96.7970),
    "Houston_TX":       (29.7604, -95.3698),
    "Chicago_IL":       (41.8781, -87.6298),
    "Memphis_TN":       (35.1495, -90.0490),
    "Atlanta_GA":       (33.7490, -84.3880),
    "Nashville_TN":     (36.1627, -86.7816),
    "Indianapolis_IN":  (39.7684, -86.1581),
    "Louisville_KY":    (38.2527, -85.7585),
    "St_Louis_MO":      (38.6270, -90.1994),
    "Oklahoma_City_OK": (35.4676, -97.5164),
    "Kansas_City_MO":   (39.0997, -94.5786),
    "Little_Rock_AR":   (34.7465, -92.2896),
    "Jackson_MS":       (32.2988, -90.1848),
    "Birmingham_AL":    (33.5207, -86.8025),
    "Tulsa_OK":         (36.1540, -95.9928),
}

CITY_NAMES = list(CITIES.keys())


def jitter(lat, lon, radius_km=1.0):
    """Add small random offset to simulate different parking locations."""
    dlat = random.gauss(0, radius_km / 111.0)
    dlon = random.gauss(0, radius_km / (111.0 * math.cos(math.radians(lat))))
    return round(lat + dlat, 6), round(lon + dlon, 6)


def interpolate_route(start, end, num_points):
    """Generate intermediate GPS pings along a route."""
    points = []
    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        lat = start[0] + t * (end[0] - start[0]) + random.gauss(0, 0.01)
        lon = start[1] + t * (end[1] - start[1]) + random.gauss(0, 0.01)
        points.append((round(lat, 6), round(lon, 6)))
    return points


def make_driver_profile(driver_id):
    """Create a driver with consistent, predictable habits."""
    home_city = random.choice(CITY_NAMES)
    # Bedtimes clustered between 21:00-23:00 (no one past midnight)
    preferred_hour = 21.0 + random.random() * 2.0  # uniform [21.0, 23.0]
    return {
        "driver_id": f"DRV-{driver_id:03d}",
        "home_city": home_city,
        "preferred_sleep_hour": preferred_hour,
        "sleep_duration_mean": random.uniform(6.5, 8.0),
        "sleep_duration_std": random.uniform(0.15, 0.4),  # tight: 10-25 min std
        "weekend_rest_prob": random.uniform(0.5, 0.8),
        "drive_days_per_week": random.randint(4, 6),
    }


def generate_driver_data(profile):
    """Generate a full year of GPS pings and sleep labels for one driver."""
    rows = []
    current_date = START_DATE
    current_city = profile["home_city"]

    while current_date < END_DATE:
        day_of_week = current_date.weekday()
        day_of_year = current_date.timetuple().tm_yday

        # Seasonal adjustment: small shift (sleep ~20 min longer in winter)
        season_factor = 0.3 * math.sin(2 * math.pi * (day_of_year - 80) / 365)

        # Weekend rest
        is_rest_day = False
        if day_of_week >= 5 and random.random() < profile["weekend_rest_prob"]:
            is_rest_day = True

        # Rare mid-week rest day (~3%)
        if not is_rest_day and random.random() < 0.03:
            is_rest_day = True

        if is_rest_day:
            lat, lon = jitter(*CITIES[current_city])
            for hour in [8, 12, 16, 19]:
                ts = current_date + timedelta(hours=hour, minutes=random.randint(0, 30))
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "driver_id": profile["driver_id"],
                    "latitude": lat,
                    "longitude": lon,
                    "speed_mph": 0.0,
                    "engine_on": False,
                    "event": "stationary",
                    "location_label": current_city,
                    "is_sleeping": False,
                })

            # Sleep: very close to preferred time on rest days
            sleep_hour = profile["preferred_sleep_hour"] + random.gauss(0, 0.2)
            sleep_dur = max(5.0, profile["sleep_duration_mean"] - season_factor * 0.3
                           + random.gauss(0, profile["sleep_duration_std"]))

            sleep_start = current_date + timedelta(hours=sleep_hour)
            sleep_end = sleep_start + timedelta(hours=sleep_dur)

            rows.append({
                "timestamp": sleep_start.strftime("%Y-%m-%d %H:%M:%S"),
                "driver_id": profile["driver_id"],
                "latitude": lat,
                "longitude": lon,
                "speed_mph": 0.0,
                "engine_on": False,
                "event": "sleep_start",
                "location_label": current_city,
                "is_sleeping": True,
            })
            rows.append({
                "timestamp": sleep_end.strftime("%Y-%m-%d %H:%M:%S"),
                "driver_id": profile["driver_id"],
                "latitude": lat,
                "longitude": lon,
                "speed_mph": 0.0,
                "engine_on": False,
                "event": "sleep_end",
                "location_label": current_city,
                "is_sleeping": False,
            })

        else:
            # --- Driving day ---
            dest_city = random.choice([c for c in CITY_NAMES if c != current_city])
            start_coords = CITIES[current_city]
            end_coords = CITIES[dest_city]

            # Wake up based on sleep pattern
            wake_hour = profile["preferred_sleep_hour"] + profile["sleep_duration_mean"] - 24
            wake_hour = max(5.0, wake_hour + random.gauss(0, 0.2) - season_factor * 0.15)
            depart_hour = wake_hour + random.uniform(0.5, 1.0)

            wake_lat, wake_lon = jitter(*start_coords)
            wake_ts = current_date + timedelta(hours=wake_hour)

            rows.append({
                "timestamp": wake_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "driver_id": profile["driver_id"],
                "latitude": wake_lat,
                "longitude": wake_lon,
                "speed_mph": 0.0,
                "engine_on": False,
                "event": "wake_up",
                "location_label": current_city,
                "is_sleeping": False,
            })

            # Driving pings
            drive_duration_hours = random.uniform(7, 10)
            num_pings = int(drive_duration_hours * random.uniform(1.2, 1.8))
            route_points = interpolate_route(start_coords, end_coords, num_pings)

            for i, (lat, lon) in enumerate(route_points):
                t_offset = depart_hour + (drive_duration_hours * i / max(len(route_points) - 1, 1))
                ts = current_date + timedelta(hours=t_offset, minutes=random.randint(0, 10))
                speed = random.uniform(50, 70) if i > 0 else random.uniform(10, 25)

                is_break = (i == len(route_points) // 2) and random.random() < 0.85
                if is_break:
                    speed = 0.0

                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "driver_id": profile["driver_id"],
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "speed_mph": round(speed, 1),
                    "engine_on": not is_break,
                    "event": "driving_break" if is_break else "driving",
                    "location_label": "",
                    "is_sleeping": False,
                })

            # Arrive
            arrive_hour = depart_hour + drive_duration_hours
            arr_lat, arr_lon = jitter(*end_coords)
            arrive_ts = current_date + timedelta(hours=arrive_hour)

            rows.append({
                "timestamp": arrive_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "driver_id": profile["driver_id"],
                "latitude": arr_lat,
                "longitude": arr_lon,
                "speed_mph": 0.0,
                "engine_on": False,
                "event": "arrive",
                "location_label": dest_city,
                "is_sleeping": False,
            })

            # Evening sleep - mild jitter around preferred time
            sleep_hour = profile["preferred_sleep_hour"] + random.gauss(0, 0.3)
            # If arrived late, push sleep a bit but not wildly
            if arrive_hour > sleep_hour:
                sleep_hour = arrive_hour + random.uniform(0.3, 0.7)

            sleep_dur = max(5.0, profile["sleep_duration_mean"] - season_factor * 0.3
                           + random.gauss(0, profile["sleep_duration_std"]))

            sleep_start = current_date + timedelta(hours=sleep_hour)
            sleep_end = sleep_start + timedelta(hours=sleep_dur)

            rows.append({
                "timestamp": sleep_start.strftime("%Y-%m-%d %H:%M:%S"),
                "driver_id": profile["driver_id"],
                "latitude": arr_lat,
                "longitude": arr_lon,
                "speed_mph": 0.0,
                "engine_on": False,
                "event": "sleep_start",
                "location_label": dest_city,
                "is_sleeping": True,
            })
            rows.append({
                "timestamp": sleep_end.strftime("%Y-%m-%d %H:%M:%S"),
                "driver_id": profile["driver_id"],
                "latitude": arr_lat,
                "longitude": arr_lon,
                "speed_mph": 0.0,
                "engine_on": False,
                "event": "sleep_end",
                "location_label": dest_city,
                "is_sleeping": False,
            })

            current_city = dest_city

        current_date += timedelta(days=1)

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    driver_profiles = []

    for i in range(1, NUM_DRIVERS + 1):
        profile = make_driver_profile(i)
        driver_profiles.append(profile)
        print(f"Generating data for {profile['driver_id']} "
              f"(home: {profile['home_city']}, "
              f"bedtime: {profile['preferred_sleep_hour']:.1f}h, "
              f"sleep: {profile['sleep_duration_mean']:.1f}h +/- {profile['sleep_duration_std']:.2f}h)")
        rows = generate_driver_data(profile)
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: r["timestamp"])

    fields = ["timestamp", "driver_id", "latitude", "longitude", "speed_mph",
              "engine_on", "event", "location_label", "is_sleeping"]

    filepath = os.path.join(OUTPUT_DIR, "driver_gps_events.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows):,} rows to {filepath}")

    prof_fields = ["driver_id", "home_city", "preferred_sleep_hour",
                   "sleep_duration_mean", "sleep_duration_std",
                   "weekend_rest_prob", "drive_days_per_week"]
    prof_path = os.path.join(OUTPUT_DIR, "driver_profiles.csv")
    with open(prof_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prof_fields)
        writer.writeheader()
        for p in driver_profiles:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in p.items() if k in prof_fields})
    print(f"Wrote {len(driver_profiles)} driver profiles to {prof_path}")

    sleep_rows = []
    sleep_starts = {}
    for row in all_rows:
        if row["event"] == "sleep_start":
            sleep_starts[row["driver_id"]] = row
        elif row["event"] == "sleep_end" and row["driver_id"] in sleep_starts:
            start = sleep_starts.pop(row["driver_id"])
            s = datetime.strptime(start["timestamp"], "%Y-%m-%d %H:%M:%S")
            e = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            duration_hours = (e - s).total_seconds() / 3600
            sleep_rows.append({
                "driver_id": start["driver_id"],
                "sleep_start": start["timestamp"],
                "sleep_end": row["timestamp"],
                "duration_hours": round(duration_hours, 2),
                "latitude": start["latitude"],
                "longitude": start["longitude"],
                "location_label": start["location_label"],
                "day_of_week": s.strftime("%A"),
                "date": s.strftime("%Y-%m-%d"),
            })

    sleep_path = os.path.join(OUTPUT_DIR, "sleep_events.csv")
    sleep_fields = ["driver_id", "date", "day_of_week", "sleep_start", "sleep_end",
                    "duration_hours", "latitude", "longitude", "location_label"]
    with open(sleep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sleep_fields)
        writer.writeheader()
        writer.writerows(sleep_rows)
    print(f"Wrote {len(sleep_rows):,} sleep events to {sleep_path}")


if __name__ == "__main__":
    main()
