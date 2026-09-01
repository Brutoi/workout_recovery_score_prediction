# %%
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

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

def historical_feature_generation(df):
    df1 = df.copy()
    
    df1["date"] = pd.to_datetime(df1["date"])
    df1 = df1.sort_values(["user_id", "date"]).copy()
    
    df1["high_intensity_min"] = df1["hr_zone_3_min"] + df1["hr_zone_4_min"] + df1["hr_zone_5_min"]
    df1["low_intensity_min"] = df1["hr_zone_1_min"] + df1["hr_zone_2_min"]
    df1["hrv_ratio"] = df1["hrv"] / df1["hrv_baseline"]
    df1["rhr_ratio"] = df1["resting_heart_rate"] / df1["rhr_baseline"]
    df1["rem_sleep_hours_ratio"] = df1["rem_sleep_hours"] / df1["sleep_hours"]
    df1["deep_sleep_hours_ratio"] = df1["deep_sleep_hours"] / df1["sleep_hours"]
    df1["bmi"] = (df1["weight_kg"] / (df1["height_cm"] / 100) ** 2)
    
    #===============================================================
    
    df1["month"] = df1["date"].dt.month

    df1["month_sin"] = np.sin(2 * np.pi * df1["month"] / 12)
    df1["month_cos"] = np.cos(2 * np.pi * df1["month"] / 12)

    df1["day_of_week_num"] = df1["date"].dt.dayofweek

    df1["day_of_week_sin"] = np.sin(
        2 * np.pi * df1["day_of_week_num"] / 7
    )

    df1["day_of_week_cos"] = np.cos(
        2 * np.pi * df1["day_of_week_num"] / 7
    )

    df1["is_weekend"] = (df1["day_of_week_num"] >= 5).astype(int)
    
    #===============================================================
    
    df1['sleep_quality_score'] = (
        df1['sleep_efficiency'] * 0.4 +
        (df1['deep_sleep_hours'] / df1['sleep_hours'].clip(lower=0.1)) * 30 +
        (1 / (df1['wake_ups'] + 1)) * 20
    ).clip(0, 100)
    
    df1['hrv_rhr_ratio'] = (
        df1['hrv'] /
        (df1['resting_heart_rate'] + 1)
    )
    
    df1['hrv_deviation'] = (
        df1['hrv'] - df1['hrv_baseline']
    )

    df1['rhr_deviation'] = (
        df1['resting_heart_rate'] -
        df1['rhr_baseline']
    )
    
    df1["strain_sleep_ratio"] = (
        df1["day_strain"] /
        (df1["sleep_hours"] + 1)
    )
    
    #===============================================================
    
    rolling_metrics = [
        "day_strain",
        "sleep_hours",
        "hrv"
    ]
    
    for col in rolling_metrics:
        df1[f"{col}_rolling_mean_7d"] = (
            df1.groupby("user_id")[col]
            .transform(
                lambda x: x.shift(1)
                        .rolling(window=7, min_periods=1)
                        .mean()
            )
        )
    
    #==============================================================
    
    historical_cols = [
        "recovery_score",
        "day_strain",
        "sleep_hours",
        "sleep_efficiency",
        "hrv",
        "resting_heart_rate",
        "calories_burned"
    ]

    for col in historical_cols:

        df1[f"previous_{col}"] = (
            df1.groupby("user_id")[col]
            .shift(1)
        )
        
    #==============================================================
    
    for col in [
        "day_strain",
        "sleep_hours",
        "sleep_efficiency",
        "hrv",
        "resting_heart_rate"
    ]:
        df1[f"{col}_change"] = (
            df1[col] -
            df1[f"previous_{col}"]
        )
        

    #==============================================================
    
    rolling_cols = [
            "day_strain",
            "sleep_hours",
            "sleep_efficiency",
            "hrv",
            "resting_heart_rate"
        ]

    for col in rolling_cols:

        df1[f"{col}_rolling_mean_7d"] = (
            df1.groupby("user_id")[col]
            .transform(
                lambda x:
                x.shift(1)
                 .rolling(window=7, min_periods=1)
                 .mean()
            )
        )
        
    #==============================================================
        
    for col in [
        "day_strain",
        "sleep_hours",
        "hrv",
        "resting_heart_rate"
    ]:

        user_mean = (
            df1.groupby("user_id")[col]
            .transform(
                lambda x:
                x.shift(1).expanding().mean()
            )
        )

        df1[f"{col}_vs_personal_mean"] = (
            df1[col] - user_mean
        )
        
    #==============================================================
    
    for col in ["hrv", "resting_heart_rate"]:
        df1[f"{col}_rolling_std_7d"] = (
            df1.groupby("user_id")[col]
            .transform(
                lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
            )
            .fillna(0)  # Handle NaN on single-element windows
        )
    
    df1['hrv_zscore'] = (df1['hrv'] - df1['hrv_rolling_mean_7d']) / (
        df1['hrv_rolling_std_7d'] + 1e-5
    )
    df1['rhr_zscore'] = (
        df1['resting_heart_rate'] - df1['resting_heart_rate_rolling_mean_7d']
    ) / (df1['resting_heart_rate_rolling_std_7d'] + 1e-5)

    df1['hrv_sleep_interaction'] = df1['hrv_ratio'] * df1['sleep_performance']
    
    #==============================================================
    
    df1["previous_recovery_score"] = (
        df1.groupby("user_id")["recovery_score"].shift(1)
    )
    



        
    df1["is_first_record"] = df1["previous_hrv"].isna().astype(int)
    
    df1 = df1.drop(columns=["recovery_score"])

    return df1

# %%
def daily_feature_generation(df):
    df1 = df.copy()
    
    df1["high_intensity_min"] = df1["hr_zone_3_min"] + df1["hr_zone_4_min"] + df1["hr_zone_5_min"]
    df1["low_intensity_min"] = df1["hr_zone_1_min"] + df1["hr_zone_2_min"]
    df1["hrv_ratio"] = df1["hrv"] / df1["hrv_baseline"]
    df1["rhr_ratio"] = df1["resting_heart_rate"] / df1["rhr_baseline"]
    df1["rem_sleep_hours_ratio"] = df1["rem_sleep_hours"] / df1["sleep_hours"]
    df1["deep_sleep_hours_ratio"] = df1["deep_sleep_hours"] / df1["sleep_hours"]
    df1["bmi"] = df1["weight_kg"] / (df1["height_cm"] / 100) ** 2

    return df1

# %%
def create_temp_strat_bin(df):
    df1 = df.copy()
    df1['score_bins'] = pd.qcut(df['recovery_score'], q=10, labels=False, duplicates='drop')
    return df1


def time_based_split(df, group_col="user_id", date_col="date", test_size=0.2):
    train_frames, test_frames = [], []
    for uid, group in df.groupby(group_col):
        group = group.sort_values(date_col)
        split_point = int(len(group) * (1 - test_size))
        train_frames.append(group.iloc[:split_point])
        test_frames.append(group.iloc[split_point:])
    return pd.concat(train_frames), pd.concat(test_frames)


# %%
print("functions.py imported successfully")


