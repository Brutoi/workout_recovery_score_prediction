# %%
import numpy as np
import pandas as pd

df = pd.read_csv("../data/whoop_fitness_dataset_100k.csv")
# %%
def drop_irr_features(df):  
    col_to_drop = [
        
        "user_id",
        "date",
        "day_of_week",
        "primary_sport",
        "workout_time_of_day"
    ]
    df1 = df.copy();
    df1.drop(columns=col_to_drop, inplace=True)
    return df1


# %%
def feature_generation(df):
    df1 = df.copy()
    df1["high_intensity_min"] = df["hr_zone_3_min"] + df["hr_zone_4_min"] + df["hr_zone_5_min"]
    df1["low_intensity_min"] = df["hr_zone_1_min"] + df["hr_zone_2_min"]
    df1["hrv_ratio"] = df["hrv"] / df["hrv_baseline"]
    df1["rhr_ratio"] = df["resting_heart_rate"] / df["rhr_baseline"]
    df1["rem_sleep_hours_ratio"] = df["rem_sleep_hours"] / df["sleep_hours"]
    df1["deep_sleep_hours_ratio"] = df["deep_sleep_hours"] / df["sleep_hours"]
    df1["bmi"] = df["weight_kg"] / np.exp2(df["height_cm"] / 100)
    return df1

def create_temp_strat_bin(df):
    df1 = df.copy()
    df1['score_bins'] = pd.qcut(df['recovery_score'], q=10, labels=False, duplicates='drop')
    return df1

print("function.py imported successfully")