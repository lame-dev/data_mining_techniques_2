import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker
import datetime

# Assuming train_df and test_df are already defined as pandas DataFrames
train_data = train_df.copy()
test_data = test_df.copy()

# Select features and prepare the target
features = train_data.drop(['srch_id', 'booking_bool', 'click_bool', 'position'], axis=1)
train_data['target'] = train_data['booking_bool'] * 5 + train_data['click_bool']

# Create a GroupShuffleSplit object
splitter = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=42)

# Perform the split based on 'srch_id'
train_indices, val_indices = next(splitter.split(train_data, groups=train_data['srch_id']))

train_split = train_data.iloc[train_indices]
val_split = train_data.iloc[val_indices]

X_train = train_split[features.columns]
y_train = train_split['target']

X_val = val_split[features.columns]
y_val = val_split['target']

# Create groups for ranking
group_train = train_split.groupby('srch_id').size().to_numpy()
group_val = val_split.groupby('srch_id').size().to_numpy()

# Define the XGBRanker
ranker = XGBRanker(
    objective='rank:ndcg',
    learning_rate=0.3,
    max_depth=4,
    min_child_weight=3,
    n_estimators=300,
    early_stopping_rounds=20
)

# Train the model
ranker.fit(
    X_train, y_train,
    group=group_train,
    eval_set=[(X_val, y_val)],
    eval_group=[group_val],
    eval_metric='ndcg@5',
    verbose=True
)

# Prepare the test data
X_test = test_data[features.columns]

# Make predictions
test_data['predictions'] = ranker.predict(X_test)

# only keep srch_id and prop_id columns
test_data = test_data[['srch_id', 'prop_id', 'predictions']]

#convert srch_id and prop_id to integers
test_data['srch_id'] = test_data['srch_id'].astype(int)
test_data['prop_id'] = test_data['prop_id'].astype(int)

# Sort the predictions for each 'srch_id'
test_data = test_data.sort_values(by=['srch_id', 'predictions'], ascending=[True, False])

# Create the submission file
submission = test_data[['srch_id', 'prop_id']]
# save the submission file with datetime
now = datetime.now()
now_str = now.strftime("%Y%m%d_%H%M%S")
submission.to_csv(f'Predictions/ranked_hotels_{now_str}.csv', index=False)

