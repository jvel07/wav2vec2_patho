import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


def split_depisda_corpus(data):
    encoder = LabelEncoder()
    data['file_prefix'] = data['filename'].str[:6]
    data['file_prefix_enc'] = encoder.fit_transform(data['file_prefix'])
    n_grupos = data['file_prefix_enc'].values
    X = data.drop(columns=['label'])
    y = data['label']
    gss_train = GroupShuffleSplit(n_splits=1, random_state=42, test_size=0.30, train_size=0.70)
    gss_dev_test = GroupShuffleSplit(n_splits=1, random_state=42, test_size=0.66, train_size=0.34)

    for i, (train_index, temp_index) in enumerate(gss_train.split(X=X, y=y, groups=n_grupos)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}, group={n_grupos[train_index]}")
        print(f"  Test:  index={temp_index}, group={n_grupos[temp_index]}")
        x_train, x_temp, y_train, y_temp = X.iloc[train_index], X.iloc[temp_index], y.iloc[train_index], y.iloc[
            temp_index]

    train_df = pd.concat([x_train, y_train], axis=1)

    rest_groups = x_temp['file_prefix_enc'].values
    for i, (dev_index, test_index) in enumerate(gss_dev_test.split(X=x_temp, y=y_temp, groups=rest_groups)):
        print(f"Fold {i}:")
        print(f"  Dev: index={dev_index}, group={rest_groups[dev_index]}")
        print(f"  Test:  index={test_index}, group={rest_groups[test_index]}")
        x_dev, x_test, y_dev, y_test = x_temp.iloc[dev_index], x_temp.iloc[test_index], y_temp.iloc[dev_index], \
        y_temp.iloc[test_index]

    dev_df = pd.concat([x_dev, y_dev], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    try:
        train_df.to_csv('../metadata/depression/depured_depression_train.csv', sep=',', index=False)
        dev_df.to_csv('../metadata/depression/depured_depression_dev.csv', sep=',', index=False)
        test_df.to_csv('../metadata/depression/depured_depression_test.csv', sep=',', index=False)
        print("CSV files saved successfully!")
    except Exception as e:
        print("Error saving the csv files: ", e)

    return train_df, dev_df, test_df


if __name__ == '__main__':

    label_dir = "../metadata/depression"

    data = pd.read_csv(f"{label_dir}/depured_complete_depisda16k_chunked_4secs.csv", encoding="utf-8")
    random_numbers = np.random.randint(1, 18, size=data['label'].isna().sum())  # to fill HC NaN values
    data.loc[data['label'].isna(), 'label'] = random_numbers
    data['label'] = data['label'].astype(int)

    # strat group by filename
    split_depisda_corpus(data)
