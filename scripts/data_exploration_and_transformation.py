from collections import Counter

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

if __name__ == '__main__':

    # load dataset
    data = joblib.load("../data/fraud_dataset.pkl")

    # unpack dictionary columns in dataframe
    df_registration = data['firstRegistrationDate'].apply(pd.Series)
    df_ad_log = data['adlog_features'].apply(pd.Series)

    # create new columns for unpacked dictionary values
    for col in df_registration.columns:
        if col == 0:
            continue
        data[col + '_first_reg'] = df_registration[col]

    for col in df_ad_log:
        if col == 0:
            continue
        data[col + '_adlog'] = df_ad_log[col]

    # values_side_id = list(set(data["siteId"].to_list())) -> is just 1 value ('GERMANY')

    # Drop NaN only, Single Value only & dictionary columns
    data = data.drop(
        columns=['adlog_features', 'source_adlog', 'firstRegistrationDate', 'processingSite_adlog', "siteId"], axis=1)

    # Transform all columns to integers (Booleans & Strings that only contain numbers)
    for col in data.columns:
        x = data[col].dtypes.name
        if is_datetime(data[col]) or data[col].dtypes.name == 'bool' or col == 'makeId':
            data[col] = data[col].astype(np.int64)

    values_side_id = list(set(data["damageUnrepaired"].to_list()))
    # Transform "damageUnrepaired" to int, but only for values not NaN
    data["damageUnrepaired"] = data["damageUnrepaired"].apply(
        lambda entry: int(entry) if type(entry) is bool else entry)

    data['contact_info'] = data['contact_info'].apply(lambda x: int(x.replace('domain_', '')))
    # Encode all Object Labels via Sklearn labelencoder

    print(data.info())

    # Prepare Categorical Labels for One-Hot-Encoding
    onehot_needed_features = []
    for col in data.columns:
        if data[col].dtype.name not in ['float64', 'int64'] and not col == 'rating':
            onehot_needed_features.append(col)

    # Create Onehot encoded dataframe
    data = pd.get_dummies(data, columns=onehot_needed_features, prefix=onehot_needed_features)

    # Distribution Target Variable
    distribution = Counter(data['rating'].tolist())

    # dump transformed dataframe for further processing
    joblib.dump(data, "../data/fraud_dataset_transformed.pkl")
