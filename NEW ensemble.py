import pandas as pd
import numpy as np
from datetime import datetime

# Function to normalize predictions
def min_max_normalize(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

# Load the predictions from both models
lgbm_predictions = pd.read_csv('Predictions/premerge_LGBM_ranked_hotels_20240515_122117.csv')
xgb_predictions = pd.read_csv('Predictions/premerge_XGB_ranked_hotels_20240515_121110.csv')

# Normalize predictions for both models
lgbm_predictions = min_max_normalize(lgbm_predictions, 'predictions')
xgb_predictions = min_max_normalize(xgb_predictions, 'predictions')

# Merge the predictions on 'srch_id' and 'prop_id'
merged_predictions = xgb_predictions.merge(
    lgbm_predictions,
    on=['srch_id', 'prop_id'],
    suffixes=('_xgb', '_lgbm')
)

# Average the predictions
merged_predictions['predictions'] = (merged_predictions['predictions_xgb'] + merged_predictions['predictions_lgbm']) / 2

# Keep only the required columns
blended_predictions = merged_predictions[['srch_id', 'prop_id', 'predictions']]

# Sort the predictions for each 'srch_id'
blended_predictions = blended_predictions.sort_values(by=['srch_id', 'predictions'], ascending=[True, False])
blended_predictions_csv = blended_predictions[['srch_id', 'prop_id']]

# Save the blended predictions to a new CSV file with date time
now = datetime.now()
now_str = now.strftime("%Y%m%d_%H%M%S")
blended_predictions_csv.to_csv(f'Predictions/blended_ranked_hotels_{now_str}.csv', index=False)

print(f"Blended predictions saved to 'blended_ranked_hotels_{now_str}.csv'")

