import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime

# Load your dataset
train_data = train_df.copy()
test_data = test_df.copy()

# Select features and prepare the target
features = train_data.drop(['srch_id', 'booking_bool', 'click_bool', 'position'], axis=1)

# Constructing a combined weighted target variable
# Assign a score of 5 for bookings and 1 for clicks (if not booked)
train_data['target'] = train_data['booking_bool'] * 5 + (1 - train_data['booking_bool']) * train_data['click_bool']

X = features
y = train_data['target']

# Split data into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix format which is used by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test_data[features.columns])

# Delete the original dataframes to free up memory
del train_data, X, y, X_train, X_val, y_train, y_val

# Set up parameters for XGBoost with the objective set for LambdaMART
params = {
    'objective': 'rank:pairwise',  # Use 'rank:pairwise' for pairwise loss minimization (this uses lambdaMART)
    'eval_metric': 'ndcg',         # This is for NDCG maximization
    'eta': 0.1,
    'max_depth': 5,
    'min_child_weight': 4,
    'subsample': 0.8,
    'lambda': 2.0,
    'alpha': 0.5,
    'verbose': 1
    #'tree_method': 'hist'  # Use histogram-based method for constructing decision trees
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=300, evals=[(dtrain, 'train'), (dval, 'eval')], early_stopping_rounds=20)

# Predict rankings on the test set
test_predictions = bst.predict(dtest)

# Assigning scores to test data and sorting
test_data['score'] = test_predictions
ranked_output = test_data.sort_values(['srch_id', 'score'], ascending=[True, False])

now = datetime.now()
now_str = now.strftime("%Y%m%d_%H%M%S")
filename = f'ranked_hotels_{now_str}.csv'

# Convert 'srch_id' and 'prop_id' to integer and output the ranked properties per search query
ranked_output['srch_id'] = ranked_output['srch_id'].astype(int)
ranked_output['prop_id'] = ranked_output['prop_id'].astype(int)
ranked_output[['srch_id', 'prop_id']].to_csv(filename, index=False, header=True)

#save the model

model_filename = f'model_{now_str}.model'
bst.save_model(model_filename)

# Get feature importance
feature_importance = bst.get_score(importance_type='gain')
for feature, importance in feature_importance.items():
    print(f"Feature: {feature}, Importance: {importance}")