import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

train_data = train.copy()
test_data = test.copy()

####### tijdelijke stappen
#for now, drop categorical features
train_data = train_data.drop(['date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'srch_destination_id'], axis=1)
test_data = test_data.drop(['date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'srch_destination_id'], axis=1)

#print columns of X_train
print(X_train.columns)

feature_columns = ['prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool','prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price', 'price_usd', 'promotion_flag','srch_query_affinity_score', 'orig_destination_distance', 'comp_rate',
       'comp_inv', 'prop_review_score_log_price', 'ump', 'per_fee', 'score2ma','total_fee', 'score1d2', 'starrating_diff', 'price_diff',
       'price_usd_norm_by_srch_id', 'price_usd_norm_by_srch_destination_id','price_usd_norm_by_prop_log_historical_price',
       'price_usd_norm_by_srch_booking_window','prop_review_score_norm_by_srch_id','prop_review_score_norm_by_srch_destination_id',
       'prop_review_score_norm_by_prop_log_historical_price','prop_review_score_norm_by_srch_booking_window',
       'starrating_diff_norm_by_srch_id','starrating_diff_norm_by_srch_destination_id',
       'starrating_diff_norm_by_prop_log_historical_price','starrating_diff_norm_by_srch_booking_window', 'average_position',
       'prob_book', 'prob_click', 'price_usd_rank','prop_location_score1_rank', 'prop_location_score2_rank', 'prop_count',
       'prop_click_count', 'prop_book_count', 'ctr', 'cvr','prop_avg_price_usd', 'prop_avg_starrating', 'comp_rate_mean',
       'comp_inv_mean']

# Select features and prepare the target
features = train_data.drop(['srch_id', 'booking_bool', 'click_bool', 'position'], axis=1)
train_data['target'] = train_data['booking_bool'] * 5 + train_data['click_bool']

# Create a GroupShuffleSplit object
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

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
    learning_rate=0.2,
    max_depth=4,
    min_child_weight=3,
    n_estimators=200,
    early_stopping_rounds=20,
    random_state=42
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

# Save the model with datetime
now = datetime.now()
now_str = now.strftime("%Y%m%d_%H%M%S")
ranker.save_model(f'Models/XGB_model_{now_str}.json')

# Prepare the test data
X_test = test_data[features.columns]

# Make predictions
test_data['predictions'] = ranker.predict(X_test)

# only keep srch_id and prop_id columns
test_data = test_data[['srch_id', 'prop_id', 'predictions']]
test_data_csv = test_data.copy()
test_data_csv.to_csv(f'Predictions/premerge_XGB_ranked_hotels_{now_str}.csv', index=False)

# Convert srch_id and prop_id to integers
test_data['srch_id'] = test_data['srch_id'].astype(int)
test_data['prop_id'] = test_data['prop_id'].astype(int)

# Sort the predictions for each 'srch_id'
test_data = test_data.sort_values(by=['srch_id', 'predictions'], ascending=[True, False])

# Create the submission file
submission = test_data[['srch_id', 'prop_id']]
# save the submission file with datetime
now = datetime.now()
now_str = now.strftime("%Y%m%d_%H%M%S")
submission.to_csv(f'Predictions/XGB_ranked_hotels_{now_str}.csv', index=False)

#feature importance
feature_importance = ranker.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features.columns)
plt.show()

