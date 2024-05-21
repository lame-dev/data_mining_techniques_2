import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from lightgbm import LGBMRanker
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

test_data = test.copy()
test_data = test_data.drop(['date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'srch_destination_id'], axis=1)

# Feature importance, sorted by importance
importance = pd.DataFrame(ranker_lgbm.feature_importances_, index=features.columns, columns=['importance'])
importance = importance.sort_values(by='importance', ascending=False)

# Identify features with zero importance
zero_importance_features = importance[importance['importance'] == 0].index.tolist()
print("Features with zero importance:", zero_importance_features)

# Drop features with zero importance
X_train = X_train.drop(columns=zero_importance_features)
X_val = X_val.drop(columns=zero_importance_features)
X_test = test_data[features.columns].drop(columns=zero_importance_features)

# Retrain the model with the reduced set of features
ranker_lgbm.fit(
    X_train, y_train,
    group=group_train,
    eval_set=[(X_val, y_val)],
    eval_group=[group_val],
    eval_metric='ndcg@5',
)

# Save the new model with datetime
now = datetime.now()
now_str = now.strftime("%Y%m%d_%H%M%S")
ranker_lgbm.booster_.save_model(f'Models/LGBMmodel_reduced_{now_str}.txt')

# Make predictions with the reduced model
test_data['predictions'] = ranker_lgbm.predict(X_test)

# Only keep srch_id and prop_id columns
test_data = test_data[['srch_id', 'prop_id', 'predictions']]

# Convert srch_id and prop_id to integers
test_data['srch_id'] = test_data['srch_id'].astype(int)
test_data['prop_id'] = test_data['prop_id'].astype(int)

# Sort the predictions for each 'srch_id' and save with predictions for ensemble
test_data = test_data.sort_values(by=['srch_id', 'predictions'], ascending=[True, False])
test_data_csv = test_data.copy()
test_data_csv.to_csv(f'Predictions/premerge_LGBM_ranked_hotels_reduced_{now_str}.csv', index=False)

# Create the submission file
submission = test_data[['srch_id', 'prop_id']]
submission.to_csv(f'Predictions/LGBM_ranked_hotels_reduced_{now_str}.csv', index=False)

importance = pd.DataFrame(ranker_lgbm.feature_importances_, index=X_train.columns, columns=['importance'])
importance = importance.sort_values(by='importance', ascending=False)

# Plot the 20 most important features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=importance.index[:55], data=importance[:55])
plt.title('Feature Importance (Top 20)')
plt.show()



# print all features
print(importance)

