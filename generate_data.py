"""
Generate synthetic GPS + sleep data for truck drivers over ~1 year.

This dataset is intentionally harder than a toy "sleep at 22:00 every night"
simulation. It mixes day/night/split shifts, drifting habits, fatigue
dynamics, and seasonal effects so that:

  - duration_hours has real variance (std ~1.1h instead of ~0.4h)
  - sleep_start_hour wraps across midnight for some drivers
    (so the circular encoding actually matters)
  - naive baselines (mean, seasonal-naive) leave meaningful room for
    models that can learn weekly/seasonal structure and driver-specific
    signatures

Signals a good model can learn:
  - Per-driver shift type (day / night / split-midnight)
  - Weekly cycle (longer sleep Thu/Fri/Sat nights)
  - Annual seasonality (longer sleep in winter, earlier wake in summer)
  - Slow habit drift (random walk on preferred bedtime across the year)
  - Sleep debt: a short night tends to be followed by a long one
  - Fatigue from the prior 3 days of driving hours
  - Vacation windows where the schedule breaks for ~1-2 weeks
  - Weather/breakdown disruption days with unusual timings
  - Late arrivals pushing bedtime later (delivery deadline effect)
"""

import csv
import random
import os
from datetime import datetime, timedelta
import math

random.seed(42)

# --- Configuration ---
NUM_DRIVERS = 12
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

# Shift archetypes. preferred_hour_range is in 24h [0, 24) where values > 24
# are interpreted modulo 24 at the end (used for split shifts whose bedtime
# sits past midnight, e.g. 1 AM == 25).
SHIFT_TYPES = {
    "day": {
        "weight": 6,
        "hour_range": (20.5, 23.5),
        "hour_std": 0.75,
        "dur_mean_range": (7.0, 8.2),
        "dur_std": 0.8,
    },
    "night": {
        # Long-haul drivers who sleep during the day
        "weight": 3,
        "hour_range": (7.5, 10.5),
        "hour_std": 0.95,
        "dur_mean_range": (6.2, 7.5),  # day sleep tends to be shorter
        "dur_std": 1.1,
    },
    "split": {
        # Bedtime after midnight; stored as 24.5..27 and wrapped to 0.5..3
        "weight": 3,
        "hour_range": (24.5, 27.0),
        "hour_std": 0.85,
        "dur_mean_range": (6.5, 7.8),
        "dur_std": 0.95,
    },
}


def pick_shift():
    pool = []
    for name, cfg in SHIFT_TYPES.items():
        pool.extend([name] * cfg["weight"])
    return random.choice(pool)


def jitter(lat, lon, radius_km=1.0):
    dlat = random.gauss(0, radius_km / 111.0)
    dlon = random.gauss(0, radius_km / (111.0 * math.cos(math.radians(lat))))
    return round(lat + dlat, 6), round(lon + dlon, 6)


def interpolate_route(start, end, num_points):
    points = []
    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        lat = start[0] + t * (end[0] - start[0]) + random.gauss(0, 0.01)
        lon = start[1] + t * (end[1] - start[1]) + random.gauss(0, 0.01)
        points.append((round(lat, 6), round(lon, 6)))
    return points


def make_driver_profile(driver_id):
    shift = pick_shift()
    cfg = SHIFT_TYPES[shift]
    lo, hi = cfg["hour_range"]
    preferred_hour = lo + random.random() * (hi - lo)
    dlo, dhi = cfg["dur_mean_range"]

    # Random-walk drift sequence on preferred bedtime, one value per day of
    # the year. Stays within ~±45 min of baseline. Generated once per driver.
    n_days = (END_DATE - START_DATE).days + 1
    drift = [0.0]
    for _ in range(n_days - 1):
        step = random.gauss(0, 0.03)
        new = max(-0.75, min(0.75, drift[-1] + step))
        drift.append(new)

    # A one- to two-week vacation window somewhere in the year
    vac_start = random.randint(30, n_days - 30)
    vac_len = random.randint(7, 14)

    # Day indexes for long-haul trips (~every 18-24 days)
    long_haul_days = set()
    d = random.randint(10, 21)
    while d < n_days - 2:
        long_haul_days.add(d)
        d += random.randint(18, 24)

    return {
        "driver_id": f"DRV-{driver_id:03d}",
        "home_city": random.choice(CITY_NAMES),
        "shift_type": shift,
        "preferred_sleep_hour": preferred_hour,  # may be >= 24 for split shifts
        "sleep_duration_mean": random.uniform(dlo, dhi),
        "sleep_duration_std": cfg["dur_std"] * random.uniform(0.85, 1.15),
        "hour_std": cfg["hour_std"] * random.uniform(0.85, 1.15),
        "weekend_rest_prob": random.uniform(0.35, 0.75),
        "drive_days_per_week": random.randint(4, 6),
        "_drift": drift,
        "_vacation": (vac_start, vac_start + vac_len),
        "_long_haul_days": long_haul_days,
    }


def compute_sleep_params(profile, day_idx, date, state):
    """Compute (sleep_start_hour, sleep_duration, is_disrupted) for a given day.

    `state` is a mutable dict holding running driver state:
      recent_driving_hours (sum over last 3 days), last_deficit, last_debt_paid.
    """
    base_hour = profile["preferred_sleep_hour"] + profile["_drift"][day_idx]
    base_dur = profile["sleep_duration_mean"]

    dow = date.weekday()
    doy = date.timetuple().tm_yday

    # Stronger seasonality: winter peak duration +0.6h, early-summer -0.4h.
    # Phase so late-Dec / early-Jan is the longest.
    season_dur = 0.5 * math.cos(2 * math.pi * (doy - 355) / 365.25)
    # Wake is earlier in summer -> start shifts earlier too (daylight cue)
    season_hour = -0.35 * math.cos(2 * math.pi * (doy - 172) / 365.25)

    # Weekly cycle: Thu/Fri/Sat nights longer + slightly later
    dow_dur = [0.0, 0.0, 0.05, 0.25, 0.45, 0.55, 0.25][dow]
    dow_hour = [0.0, 0.0, 0.05, 0.15, 0.35, 0.40, 0.20][dow]

    # Fatigue: extra hour of driving over 21h/3days -> +6 min sleep
    fatigue_extra = max(0.0, state.get("recent_driving_hours", 21) - 21) * 0.10

    # Sleep debt: last night's deficit (positive if short) is partially repaid
    deficit = state.get("last_deficit", 0.0)
    debt_repay = min(1.5, max(0.0, deficit) * 0.55)
    if deficit < -0.5:
        debt_repay = max(-0.8, deficit * 0.25)  # if overshot, trim next night

    # Long-haul aftermath: the day after a long haul gets +1.2h sleep, later start
    haul_extra_dur = 0.0
    haul_extra_hour = 0.0
    if (day_idx - 1) in profile["_long_haul_days"]:
        haul_extra_dur = random.uniform(0.9, 1.6)
        haul_extra_hour = random.uniform(-0.5, 0.3)
    # Night before long haul: sleep a bit shorter (early start next day)
    if day_idx in profile["_long_haul_days"]:
        haul_extra_dur -= random.uniform(0.3, 0.8)
        haul_extra_hour -= random.uniform(0.2, 0.6)

    # Vacation window: later bedtime, longer sleep
    vac_start, vac_end = profile["_vacation"]
    in_vac = vac_start <= day_idx < vac_end
    vac_extra_hour = random.uniform(0.8, 2.0) if in_vac else 0.0
    vac_extra_dur = random.uniform(0.4, 1.1) if in_vac else 0.0

    # Random disruption (~2% of days): weather, breakdown, emergency load
    is_disrupted = random.random() < 0.02
    disrupt_hour = random.gauss(0, 1.8) if is_disrupted else 0.0
    disrupt_dur = random.gauss(-0.6, 1.2) if is_disrupted else 0.0

    # Per-day noise
    hour_noise = random.gauss(0, profile["hour_std"])
    dur_noise = random.gauss(0, profile["sleep_duration_std"])

    start_hour_raw = (
        base_hour + season_hour + dow_hour + haul_extra_hour
        + vac_extra_hour + disrupt_hour + hour_noise
    )
    duration = (
        base_dur + season_dur + dow_dur + fatigue_extra + debt_repay
        + haul_extra_dur + vac_extra_dur + disrupt_dur + dur_noise
    )
    duration = max(3.8, min(12.5, duration))

    # Normalize hour into [0, 24)
    start_hour = start_hour_raw % 24.0

    return start_hour, duration, is_disrupted


def generate_driver_data(profile):
    rows = []
    state = {"recent_driving_hours": 21.0, "last_deficit": 0.0}
    driving_history = []  # rolling last-3-day hours

    current_date = START_DATE
    current_city = profile["home_city"]
    day_idx = 0

    while current_date < END_DATE:
        day_of_week = current_date.weekday()

        # Rest day decision
        vac_start, vac_end = profile["_vacation"]
        in_vac = vac_start <= day_idx < vac_end

        is_rest_day = False
        if in_vac and random.random() < 0.75:
            is_rest_day = True
        elif day_of_week >= 5 and random.random() < profile["weekend_rest_prob"]:
            is_rest_day = True
        elif random.random() < 0.03:
            is_rest_day = True

        is_long_haul = day_idx in profile["_long_haul_days"] and not is_rest_day

        # Decide driving hours for today (drives downstream fatigue state)
        if is_rest_day:
            drive_hours_today = 0.0
        elif is_long_haul:
            drive_hours_today = random.uniform(11.0, 13.5)
        else:
            drive_hours_today = random.uniform(6.5, 9.5)

        sleep_hour, sleep_dur, is_disrupted = compute_sleep_params(
            profile, day_idx, current_date, state
        )

        # Emit GPS rows
        if is_rest_day:
            lat, lon = jitter(*CITIES[current_city])
            for hour in [8, 12, 16, 19]:
                ts = current_date + timedelta(hours=hour, minutes=random.randint(0, 30))
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "driver_id": profile["driver_id"],
                    "latitude": lat, "longitude": lon,
                    "speed_mph": 0.0, "engine_on": False,
                    "event": "stationary",
                    "location_label": current_city,
                    "is_sleeping": False,
                })
            sleep_lat, sleep_lon = lat, lon
            sleep_location = current_city
        else:
            dest_city = random.choice([c for c in CITY_NAMES if c != current_city])
            start_coords = CITIES[current_city]
            end_coords = CITIES[dest_city]

            # Wake time is rough: for a day-shift driver this is sleep_end of
            # the prior night, which we don't track explicitly. Just emit a
            # plausible wake event a few hours before departure.
            if profile["shift_type"] == "night":
                # Night driver wakes mid-afternoon
                wake_hour = 16.0 + random.gauss(0, 0.6)
                depart_hour = wake_hour + random.uniform(0.8, 1.5)
            elif profile["shift_type"] == "split":
                # Split driver wakes mid-morning
                wake_hour = 9.5 + random.gauss(0, 0.7)
                depart_hour = wake_hour + random.uniform(0.8, 1.5)
            else:
                wake_hour = 5.5 + random.gauss(0, 0.6)
                depart_hour = wake_hour + random.uniform(0.5, 1.2)

            wake_lat, wake_lon = jitter(*start_coords)
            wake_ts = current_date + timedelta(hours=wake_hour)
            rows.append({
                "timestamp": wake_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "driver_id": profile["driver_id"],
                "latitude": wake_lat, "longitude": wake_lon,
                "speed_mph": 0.0, "engine_on": False,
                "event": "wake_up",
                "location_label": current_city,
                "is_sleeping": False,
            })

            num_pings = int(drive_hours_today * random.uniform(1.3, 2.0))
            route_points = interpolate_route(start_coords, end_coords, num_pings)
            for i, (lat, lon) in enumerate(route_points):
                t_off = depart_hour + (
                    drive_hours_today * i / max(len(route_points) - 1, 1)
                )
                ts = current_date + timedelta(hours=t_off, minutes=random.randint(0, 10))
                speed = random.uniform(50, 72) if i > 0 else random.uniform(10, 25)
                # Longer trips have a break near the middle
                mid = len(route_points) // 2
                is_break = (i == mid) and random.random() < 0.9
                if is_break:
                    speed = 0.0
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "driver_id": profile["driver_id"],
                    "latitude": round(lat, 6), "longitude": round(lon, 6),
                    "speed_mph": round(speed, 1),
                    "engine_on": not is_break,
                    "event": "driving_break" if is_break else "driving",
                    "location_label": "",
                    "is_sleeping": False,
                })

            arrive_hour = depart_hour + drive_hours_today
            sleep_lat, sleep_lon = jitter(*end_coords)
            arrive_ts = current_date + timedelta(hours=arrive_hour)
            rows.append({
                "timestamp": arrive_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "driver_id": profile["driver_id"],
                "latitude": sleep_lat, "longitude": sleep_lon,
                "speed_mph": 0.0, "engine_on": False,
                "event": "arrive",
                "location_label": dest_city,
                "is_sleeping": False,
            })
            sleep_location = dest_city
            current_city = dest_city

            # Late-arrival effect: if the driver only got in at/after the
            # preferred bedtime, push sleep later
            base_sleep_reference = profile["preferred_sleep_hour"] % 24.0
            if profile["shift_type"] == "day" and arrive_hour > base_sleep_reference:
                delay = (arrive_hour - base_sleep_reference) + random.uniform(0.2, 0.7)
                sleep_hour = (sleep_hour + delay * 0.7) % 24.0

        # Emit sleep events. Use a synthetic datetime anchored on current_date
        # so that downstream grouping by calendar-date-of-start works.
        sleep_start_dt = current_date + timedelta(hours=sleep_hour)
        sleep_end_dt = sleep_start_dt + timedelta(hours=sleep_dur)

        rows.append({
            "timestamp": sleep_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "driver_id": profile["driver_id"],
            "latitude": sleep_lat, "longitude": sleep_lon,
            "speed_mph": 0.0, "engine_on": False,
            "event": "sleep_start",
            "location_label": sleep_location,
            "is_sleeping": True,
        })
        rows.append({
            "timestamp": sleep_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "driver_id": profile["driver_id"],
            "latitude": sleep_lat, "longitude": sleep_lon,
            "speed_mph": 0.0, "engine_on": False,
            "event": "sleep_end",
            "location_label": sleep_location,
            "is_sleeping": False,
        })

        # Update rolling state AFTER emitting today's data
        driving_history.append(drive_hours_today)
        if len(driving_history) > 3:
            driving_history.pop(0)
        state["recent_driving_hours"] = sum(driving_history)
        state["last_deficit"] = profile["sleep_duration_mean"] - sleep_dur

        current_date += timedelta(days=1)
        day_idx += 1

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    driver_profiles = []

    for i in range(1, NUM_DRIVERS + 1):
        profile = make_driver_profile(i)
        driver_profiles.append(profile)
        pref = profile["preferred_sleep_hour"] % 24.0
        print(f"{profile['driver_id']} [{profile['shift_type']:>5}]  "
              f"home={profile['home_city']:<18s}  "
              f"bedtime≈{pref:4.1f}h  "
              f"sleep={profile['sleep_duration_mean']:.1f}h "
              f"(±{profile['sleep_duration_std']:.2f}h)")
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

    # Driver profile CSV (stripped of private state fields)
    prof_fields = ["driver_id", "home_city", "shift_type",
                   "preferred_sleep_hour", "sleep_duration_mean",
                   "sleep_duration_std", "weekend_rest_prob",
                   "drive_days_per_week"]
    prof_path = os.path.join(OUTPUT_DIR, "driver_profiles.csv")
    with open(prof_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prof_fields)
        writer.writeheader()
        for p in driver_profiles:
            row = {k: p[k] for k in prof_fields}
            row["preferred_sleep_hour"] = round(row["preferred_sleep_hour"] % 24.0, 4)
            for k in ("sleep_duration_mean", "sleep_duration_std", "weekend_rest_prob"):
                row[k] = round(row[k], 4)
            writer.writerow(row)
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

    # Quick variance summary to reassure ourselves the data has signal
    import statistics
    durs = [r["duration_hours"] for r in sleep_rows]
    starts = []
    for r in sleep_rows:
        t = datetime.strptime(r["sleep_start"], "%Y-%m-%d %H:%M:%S")
        starts.append(t.hour + t.minute / 60.0)
    print(f"\nduration_hours   mean={statistics.mean(durs):.2f}  std={statistics.stdev(durs):.2f}  "
          f"range=[{min(durs):.2f}, {max(durs):.2f}]")
    print(f"sleep_start_hour mean={statistics.mean(starts):.2f}  std={statistics.stdev(starts):.2f}  "
          f"range=[{min(starts):.2f}, {max(starts):.2f}]")


if __name__ == "__main__":
    main()
