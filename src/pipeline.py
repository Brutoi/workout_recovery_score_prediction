# %%
from functions import drop_irr_features, feature_generation
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
)

set_config(transform_output="pandas")

# %%
drop_irr_feat_transformer = FunctionTransformer(drop_irr_features)
generate_feature_transformer = FunctionTransformer(feature_generation)

# %%
col_to_oneHotEncoding = ["gender", "activity_type"]
col_to_ordinalEncoding = ["fitness_level"]
fitness_level_order = ["Beginner", "Intermediate", "Advanced", "Elite"]

encoding_transformer = ColumnTransformer(
    transformers=[
        ("one hot encoding", 
         OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
         col_to_oneHotEncoding),
        ("ordinal encoding",
         OrdinalEncoder(categories=[fitness_level_order], 
                        unknown_value=-1, 
                        handle_unknown='use_encoded_value'),
         col_to_ordinalEncoding
         )
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
)

# %%
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV

estimator = LGBMRegressor(
    random_state=42,
    n_estimators=50,
    importance_type="gain",  
    verbose=-1,  
    n_jobs=1
)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

feature_selector = RFECV(
    estimator=estimator,
    step=0.1,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

# %%
data_preparation_pipeline = Pipeline([
    ('drop_irrelevant_features', drop_irr_feat_transformer),
    ('feature_generation', generate_feature_transformer),
    ('encoding', encoding_transformer),
    ('feature_selection', feature_selector)
]).set_output(transform="pandas")


print("pipeline.py imported successfully")