import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, GroupKFold
from lightgbm import LGBMRanker
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


train_data = train.copy()
test_data = test.copy()

# Drop categorical features for now
train_data = train_data.drop(['date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'srch_destination_id'], axis=1)
test_data = test_data.drop(['date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'srch_destination_id'], axis=1)

# Prepare the target
features = train_data.drop(['srch_id', 'booking_bool', 'click_bool', 'position'], axis=1)
train_data['target'] = train_data['booking_bool'] * 5 + train_data['click_bool']

# Create a GroupShuffleSplit object
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=27)

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

# Define the LGBMRanker with XGBoost-like parameters
ranker_lgbm = LGBMRanker(
    objective='lambdarank',
    learning_rate=0.1,
    max_depth=6,
    num_leaves=70,
    n_estimators=300,
    subsample=0.9,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_weight=4,
    random_state=42,
    force_col_wise=True,
    verbose=1
)

# Train the model with early stopping
ranker_lgbm.fit(
    X_train, y_train,
    group=group_train,
    eval_set=[(X_val, y_val)],
    eval_group=[group_val],
    eval_metric='ndcg@5'
)

# Prepare the test data
X_test = test_data[features.columns]

# Make predictions
test_data['predictions'] = ranker_lgbm.predict(X_test)

# Only keep srch_id and prop_id columns
test_data = test_data[['srch_id', 'prop_id', 'predictions']]

# Convert srch_id and prop_id to integers
test_data['srch_id'] = test_data['srch_id'].astype(int)
test_data['prop_id'] = test_data['prop_id'].astype(int)

# Save the model with datetime
now = datetime.now()
now_str = now.strftime("%Y%m%d_%H%M%S")
#ranker_lgbm.save_model(f'Models/XGB_model_{now_str}.json')

# Sort the predictions for each 'srch_id' and save with predictions for ensemble
test_data = test_data.sort_values(by=['srch_id', 'predictions'], ascending=[True, False])
test_data_csv = test_data.copy()
#test_data_csv.to_csv(f'Predictions/premerge_LGBM_ranked_hotels_{now_str}.csv', index=False)

# Create the submission file
submission = test_data[['srch_id', 'prop_id']]
submission.to_csv(f'Predictions/LGBM_ranked_hotels_{now_str}.csv', index=False)

# Feature importance, sorted by importance
importance = pd.DataFrame(ranker_lgbm.feature_importances_, index=features.columns, columns=['importance'])
importance = importance.sort_values(by='importance', ascending=False)

# Plot the 20 most important features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=importance.index[:20], data=importance[:20])
plt.title('Feature Importance (Top)')
plt.show()

# Plot the 20 least important features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=importance.index[-30:], data=importance[-30:], color='orange')
plt.title('Feature Importance (Bottom)')
plt.show()


## GroupKFold cross-validator
#group_kfold = GroupKFold(n_splits=5)
#
## Perform cross-validation
#cv_scores = cross_val_score(ranker_lgbm, X_train, y_train, groups=train_split['srch_id'], cv=group_kfold, scoring='ndcg_scorer')  # Use a suitable scorer for ranking
#print("Cross-validation scores: ", cv_scores)
#print("Mean cross-validation score: ", np.mean(cv_scores))

