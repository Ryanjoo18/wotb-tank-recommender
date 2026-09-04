import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

APPLICATION_ID = st.secrets["APPLICATION_ID"]

API_URL = "https://api.wotblitz.asia/wotb/encyclopedia/vehicles/"

features = [
    "tier",
    "hp",
    "avg_damage",
    "avg_penetration",
    "caliber",
    "dpm",
    "reload_time",
    "gun_elevation",
    "gun_depression",
    "gun_handling",
    "clip_capacity",
    "clip_reload_time",
    "burst_damage",
    "hull_armor_avg",
    "turret_armor_avg",
    "speed_forward",
    "speed_backward",
    "power_to_weight",
    "hull_traverse",
    "turret_traverse",
    "view_range",
]

# extract tank features from the Wargaming API response
def extract_tank_features(tanks):
    rows = []

    for tank_id, tank in tanks.items():
        profile = tank.get("default_profile", {})
        gun = profile.get("gun", {})
        turret = profile.get("turret", {})
        engine = profile.get("engine", {})
        suspension = profile.get("suspension", {})
        armor = profile.get("armor", {})
        hull_armor = armor.get("hull", {})
        turret_armor = armor.get("turret", {})
        shells = profile.get("shells", [])
        weight = profile.get("weight")
        engine_power = engine.get("power")
        power_to_weight = engine_power / weight if weight and engine_power is not None else np.nan
        damages = [shell.get("damage") for shell in shells]
        penetrations = [shell.get("penetration") for shell in shells]
        avg_damage = np.mean(damages) if damages and all(value is not None for value in damages) else np.nan
        avg_penetration = np.mean(penetrations) if penetrations and all(value is not None for value in penetrations) else np.nan

        rows.append(
            {
                "name": tank.get("name"),
                "nation": tank.get("nation"),
                "type": tank.get("type"),
                "tier": tank.get("tier"),
                "is_premium": tank.get("is_premium"),
                "image_preview": tank.get("images", {}).get("preview"),
                "image_normal": tank.get("images", {}).get("normal"),
                "hp": profile.get("hp"),
                "speed_forward": profile.get("speed_forward"),
                "speed_backward": profile.get("speed_backward"),
                "weight": weight,
                "hull_armor_front": hull_armor.get("front"),
                "hull_armor_rear": hull_armor.get("rear"),
                "hull_armor_side": hull_armor.get("sides"),
                "turret_armor_front": turret_armor.get("front"),
                "turret_armor_rear": turret_armor.get("rear"),
                "turret_armor_side": turret_armor.get("sides"),
                "engine_power": engine_power,
                "aim_time": gun.get("aim_time"),
                "caliber": gun.get("caliber"),
                "clip_capacity": gun.get("clip_capacity"),
                "clip_reload_time": gun.get("clip_reload_time"),
                "gun_depression": gun.get("move_down_arc"),
                "gun_elevation": gun.get("move_up_arc"),
                "dispersion": gun.get("dispersion"),
                "fire_rate": gun.get("fire_rate"),
                "reload_time": gun.get("reload_time"),
                "hull_traverse": suspension.get("traverse_speed"),
                "turret_traverse": turret.get("traverse_speed"),
                "view_range": turret.get("view_range"),
                "power_to_weight": power_to_weight,
                "avg_damage": avg_damage,
                "avg_penetration": avg_penetration,
            }
        )

    return pd.DataFrame(rows)

# add derived features
def add_features(df):
    df = df.copy()

    df["dpm"] = df["avg_damage"] * df["fire_rate"]
    df["hull_armor_avg"] = (df["hull_armor_front"] + df["hull_armor_side"] + df["hull_armor_rear"]) / 3
    df["turret_armor_avg"] = (df["turret_armor_front"] + df["turret_armor_side"] + df["turret_armor_rear"]) / 3
    df["burst_damage"] = df["avg_damage"] * df["clip_capacity"]
    df["gun_handling"] = (1 / df["aim_time"] * df["dispersion"]).replace([np.inf, -np.inf], np.nan)

    return df

# retrieve tank data from API
def load_tank_data():
    if not APPLICATION_ID:
        raise ValueError("API key is not configured. Add API key to the .env file.")

    params = {
        "application_id": APPLICATION_ID,
        "language": "en",
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    tanks = data.get("data")

    if not tanks:
        raise ValueError("Wargaming API returned no tank data.")

    tank_df = extract_tank_features(tanks)
    tank_df = add_features(tank_df)
    # remove tanks with missing values in any feature
    tank_df = tank_df.dropna(subset=features).reset_index(drop=True)

    return tank_df

# standardise numerical features, 
# calculate cosine similarity between tanks
def build_similarity_matrix(tank_df):
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(tank_df[features])
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix

# find tanks that are most similar to the selected tank
def recommend_tanks(tank_name,
                    tank_df,
                    similarity_matrix,
                    number_of_recommendations=10,
                    exclude_premium=False
                    ):

    # try to match tank name exactly; if fails, do partial matching
    exact_matches = tank_df[tank_df["name"].str.lower() == tank_name.lower()]

    if exact_matches.empty:
        matches = tank_df[tank_df["name"].str.contains(tank_name, case=False, na=False)]
    else:
        matches = exact_matches

    if matches.empty:
        return None, pd.DataFrame()

    selected_index = matches.index[0]
    matched_name = tank_df.loc[selected_index, "name"]

    # retrieve similarity scores for the selected tank
    similarities = similarity_matrix[selected_index]

    # sort tanks by similarity in descending order
    ranked_indices = np.argsort(similarities)[::-1]

    # remove the selected tank from the recommendations
    ranked_indices = [index for index in ranked_indices if index != selected_index]

    # optionally remove premium tanks from the recommendations
    if exclude_premium:
        ranked_indices = [index for index in ranked_indices if tank_df.iloc[index]["is_premium"] == 0]

    # keep only the requested number of recommendations
    ranked_indices = ranked_indices[:number_of_recommendations]

    recommendations = tank_df.iloc[ranked_indices].copy()

    # add the cosine similarity score to the recommendation dataframe
    recommendations["similarity"] = [similarities[index] for index in ranked_indices]

    return matched_name, recommendations