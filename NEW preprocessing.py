import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load a small portion of your datasets to get the column names
df_temp = pd.read_csv('data/training_set_VU_DM.csv', nrows=5)

# Create the dtypes dictionary
dtypes = {col: 'int32' if col in ['srch_id', 'prop_id'] else 'float32' for col in df_temp.columns if col != 'date_time'}
dtypes['date_time'] = 'object'

# Now load your full datasets with the specified data types
train = pd.read_csv('data/training_set_VU_DM.csv', dtype=dtypes)
test = pd.read_csv('data/test_set_VU_DM.csv', dtype=dtypes)

# List of competitor rate and inventory columns
comp_rate_cols = [f'comp{i}_rate' for i in range(1, 9)]
comp_inv_cols = [f'comp{i}_inv' for i in range(1, 9)]
comp_rate_percent_diff_cols = [f'comp{i}_rate_percent_diff' for i in range(1, 9)]

# Fill missing values with 0
train[comp_rate_cols] = train[comp_rate_cols].fillna(0)
train[comp_inv_cols] = train[comp_inv_cols].fillna(0)

# Create new aggregated features
train['comp_rate'] = train[comp_rate_cols].sum(axis=1)
test['comp_rate'] = test[comp_rate_cols].sum(axis=1)
train['comp_inv'] = train[comp_inv_cols].sum(axis=1)
test['comp_inv'] = test[comp_inv_cols].sum(axis=1)

# Fill other missing values with 0 or statistical measures as appropriate
train['orig_destination_distance'].fillna(0, inplace=True)
test['orig_destination_distance'].fillna(0, inplace=True)

# Handling outliers for price_usd
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df

train = cap_outliers(train, 'price_usd')
test = cap_outliers(test, 'price_usd')

# Additional missing value handling as described in the report
train['visitor_hist_adr_usd'].fillna(train['visitor_hist_adr_usd'].median(), inplace=True)
test['visitor_hist_adr_usd'].fillna(test['visitor_hist_adr_usd'].median(), inplace=True)
train['visitor_hist_adr_usd'].fillna(train['visitor_hist_adr_usd'].median(), inplace=True)
test['visitor_hist_adr_usd'].fillna(test['visitor_hist_adr_usd'].median(), inplace=True)
train['prop_review_score'].fillna(train['prop_review_score'].median(), inplace=True)
test['prop_review_score'].fillna(test['prop_review_score'].median(), inplace=True)
train['prop_log_historical_price'].replace(0, train['prop_log_historical_price'].median(), inplace=True)
test['prop_log_historical_price'].replace(0, test['prop_log_historical_price'].median(), inplace=True)
train['srch_query_affinity_score'].fillna(train['srch_query_affinity_score'].median(), inplace=True)
test['srch_query_affinity_score'].fillna(test['srch_query_affinity_score'].median(), inplace=True)

# Feature Engineering
##### test features
train['search_month'] = pd.to_datetime(train['date_time']).dt.month
test['search_month'] = pd.to_datetime(test['date_time']).dt.month
train['prop_review_score_log_price'] = train['prop_review_score'] * train['prop_log_historical_price']
test['prop_review_score_log_price'] = test['prop_review_score'] * test['prop_log_historical_price']

train['ump'] = np.exp(train['prop_log_historical_price']) - train['price_usd']
test['ump'] = np.exp(test['prop_log_historical_price']) - test['price_usd']

train['per_fee'] = train['price_usd'] * train['srch_room_count'] / (train['srch_adults_count'] + train['srch_children_count'])
test['per_fee'] = test['price_usd'] * test['srch_room_count'] / (test['srch_adults_count'] + test['srch_children_count'])

train['score2ma'] = train['prop_location_score2'] * train['srch_query_affinity_score']
test['score2ma'] = test['prop_location_score2'] * test['srch_query_affinity_score']

train['total_fee'] = train['price_usd'] * train['srch_room_count']
test['total_fee'] = test['price_usd'] * test['srch_room_count']

train['score1d2'] = (train['prop_location_score2'] + 0.0001) / (train['prop_location_score1'] + 0.0001)
test['score1d2'] = (test['prop_location_score2'] + 0.0001) / (test['prop_location_score1'] + 0.0001)

# Interaction between User and Property Historical Data
train['starrating_diff'] = train['visitor_hist_starrating'] - train['prop_starrating']
test['starrating_diff'] = test['visitor_hist_starrating'] - test['prop_starrating']

train['price_diff'] = train['visitor_hist_adr_usd'] - train['price_usd']
test['price_diff'] = test['visitor_hist_adr_usd'] - test['price_usd']

# Normalizing specific features
grouping_features = ['srch_id', 'srch_destination_id', 'prop_log_historical_price', 'srch_booking_window']

def normalize_feature(df, feature, grouping_feature):
    normalized_feature_name = f'{feature}_norm_by_{grouping_feature}'
    df[normalized_feature_name] = df[feature] / (df.groupby(grouping_feature)[feature].transform('mean') + 0.0001)
    return df

for feature in ['price_usd', 'prop_review_score', 'starrating_diff']:
    for grouping_feature in grouping_features:
        train = normalize_feature(train, feature, grouping_feature)
        test = normalize_feature(test, feature, grouping_feature)

# For demonstration, if there were categorical features, we would encode them as follows
categorical_features = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'srch_destination_id']

# Additional Preprocessing Steps

# Calculate average_position for each group in the training set, position bias
train['average_position'] = train.groupby(['prop_id', 'prop_country_id'])['position'].transform('mean')
average_position_df = train.groupby(['prop_id', 'prop_country_id'])['average_position'].mean().reset_index()
test = test.merge(average_position_df, on=['prop_id', 'prop_country_id'], how='left')

# Fill any missing values in the test set (if any) after the merge
test['average_position'].fillna(test['average_position'].mean(), inplace=True)  # or use another imputation method

# Create combined feature for adults and children
train['total_people'] = train['srch_adults_count'] + train['srch_children_count']
test['total_people'] = test['srch_adults_count'] + test['srch_children_count']

# Normalize additional features
for feature in ['price_usd', 'prop_review_score', 'starrating_diff']:
    for grouping_feature in grouping_features:
        train = normalize_feature(train, feature, grouping_feature)
        test = normalize_feature(test, feature, grouping_feature)

# Calculate booking and click probabilities
train['prob_book'] = train.groupby('prop_id')['booking_bool'].transform('mean')
train['prob_click'] = train.groupby('prop_id')['click_bool'].transform('mean')

# Create a DataFrame to hold the booking and click probabilities
booking_click_prob_df = train.groupby('prop_id').agg(
    prob_book=('booking_bool', 'mean'),
    prob_click=('click_bool', 'mean')
).reset_index()

test = test.merge(booking_click_prob_df, on='prop_id', how='left')

########### new features
#rank features
train['price_usd_rank'] = train.groupby('srch_id')['price_usd'].rank()
test['price_usd_rank'] = test.groupby('srch_id')['price_usd'].rank()

train['prop_location_score1_rank'] = train.groupby('srch_id')['prop_location_score1'].rank()
test['prop_location_score1_rank'] = test.groupby('srch_id')['prop_location_score1'].rank()

train['prop_location_score2_rank'] = train.groupby('srch_id')['prop_location_score2'].rank()
test['prop_location_score2_rank'] = test.groupby('srch_id')['prop_location_score2'].rank()

# count features
prop_count = train['prop_id'].value_counts().reset_index()
prop_count.columns = ['prop_id', 'prop_count']
train = train.merge(prop_count, on='prop_id', how='left')
test = test.merge(prop_count, on='prop_id', how='left')

click_count = train.groupby('prop_id')['click_bool'].sum().reset_index()
click_count.columns = ['prop_id', 'prop_click_count']
train = train.merge(click_count, on='prop_id', how='left')
test = test.merge(click_count, on='prop_id', how='left')

book_count = train.groupby('prop_id')['booking_bool'].sum().reset_index()
book_count.columns = ['prop_id', 'prop_book_count']
train = train.merge(book_count, on='prop_id', how='left')
test = test.merge(book_count, on='prop_id', how='left')

# user behavior features
train['ctr'] = train['prob_click'] / (train['prob_click'] + train['prob_book'] + 1e-6)
test['ctr'] = test['prob_click'] / (test['prob_click'] + test['prob_book'] + 1e-6)

train['cvr'] = train['prob_book'] / (train['prob_click'] + train['prob_book'] + 1e-6)
test['cvr'] = test['prob_book'] / (test['prob_click'] + test['prob_book'] + 1e-6)

# temporal features
train['search_week'] = pd.to_datetime(train['date_time']).dt.isocalendar().week
test['search_week'] = pd.to_datetime(test['date_time']).dt.isocalendar().week
train['is_weekend'] = pd.to_datetime(train['date_time']).dt.weekday >= 5
test['is_weekend'] = pd.to_datetime(test['date_time']).dt.weekday >= 5
train['is_weekend'] = train['is_weekend'].astype(int)
test['is_weekend'] = test['is_weekend'].astype(int)

# aggregation features
train['price_usd_mean'] = train.groupby('srch_id')['price_usd'].transform('mean')
test['price_usd_mean'] = test.groupby('srch_id')['price_usd'].transform('mean')

train['prop_review_score_mean'] = train.groupby('srch_id')['prop_review_score'].transform('mean')
test['prop_review_score_mean'] = test.groupby('srch_id')['prop_review_score'].transform('mean')

# convert datetime to month
train['date_time'] = pd.to_datetime(train['date_time'])
test['date_time'] = pd.to_datetime(test['date_time'])

# User's Previous Clicks and Bookings per Destination in the training set
train['user_clicks_per_dest'] = train.groupby(['visitor_location_country_id', 'srch_destination_id'])['click_bool'].transform('sum')
train['user_bookings_per_dest'] = train.groupby(['visitor_location_country_id', 'srch_destination_id'])['booking_bool'].transform('sum')

# Create a DataFrame to hold these values
user_behavior_df = train[['visitor_location_country_id', 'srch_destination_id', 'user_clicks_per_dest', 'user_bookings_per_dest']].drop_duplicates()

# Merge this DataFrame into the test set
test = test.merge(user_behavior_df, on=['visitor_location_country_id', 'srch_destination_id'], how='left')

# Fill any missing values in the test set (if any) after the merge
test['user_clicks_per_dest'].fillna(0, inplace=True)
test['user_bookings_per_dest'].fillna(0, inplace=True)

# User's Average Search Duration
user_avg_duration = train.groupby('visitor_location_country_id')['srch_length_of_stay'].mean().reset_index().rename(columns={'srch_length_of_stay': 'user_avg_search_duration'})
train = train.merge(user_avg_duration, on='visitor_location_country_id', how='left')
test = test.merge(user_avg_duration, on='visitor_location_country_id', how='left')

# User's Booking/Click Ratio
train['user_book_click_ratio'] = train['user_bookings_per_dest'] / (train['user_clicks_per_dest'] + 1e-6)
test['user_book_click_ratio'] = test['user_bookings_per_dest'] / (test['user_clicks_per_dest'] + 1e-6)

# Historical Performance of Properties
prop_avg_price_usd = train.groupby('prop_id')['price_usd'].mean().reset_index().rename(columns={'price_usd': 'prop_avg_price_usd'})
prop_avg_starrating = train.groupby('prop_id')['prop_starrating'].mean().reset_index().rename(columns={'prop_starrating': 'prop_avg_starrating'})

train = train.merge(prop_avg_price_usd, on='prop_id', how='left')
train = train.merge(prop_avg_starrating, on='prop_id', how='left')
test = test.merge(prop_avg_price_usd, on='prop_id', how='left')
test = test.merge(prop_avg_starrating, on='prop_id', how='left')

# Competitor Features Aggregation
train['comp_rate_mean'] = train[comp_rate_cols].mean(axis=1)
test['comp_rate_mean'] = test[comp_rate_cols].mean(axis=1)
train['comp_inv_mean'] = train[comp_inv_cols].mean(axis=1)
test['comp_inv_mean'] = test[comp_inv_cols].mean(axis=1)

# Week of Month
train['search_week_of_month'] = (pd.to_datetime(train['date_time']).dt.day - 1) // 7 + 1
test['search_week_of_month'] = (pd.to_datetime(test['date_time']).dt.day - 1) // 7 + 1

# Group Aggregated Statistics
train['mean_price_usd_by_dest'] = train.groupby('srch_destination_id')['price_usd'].transform('mean')
test['mean_price_usd_by_dest'] = test.groupby('srch_destination_id')['price_usd'].transform('mean')

train['median_starrating_by_dest'] = train.groupby('srch_destination_id')['prop_starrating'].transform('median')
test['median_starrating_by_dest'] = test.groupby('srch_destination_id')['prop_starrating'].transform('median')

# Standard Deviation and Variance
train['price_usd_std'] = train.groupby('srch_id')['price_usd'].transform('std')
test['price_usd_std'] = test.groupby('srch_id')['price_usd'].transform('std')

train['prop_review_score_var'] = train.groupby('srch_id')['prop_review_score'].transform('var')
test['prop_review_score_var'] = test.groupby('srch_id')['prop_review_score'].transform('var')

# Create the count_window_feature
train['count_window_feature'] = train['srch_room_count'] * train.groupby('srch_id')['srch_booking_window'].transform('max') + train['srch_booking_window']
test['count_window_feature'] = test['srch_room_count'] * test.groupby('srch_id')['srch_booking_window'].transform('max') + test['srch_booking_window']

train['exponential_query_affinity_score'] = np.exp(train['srch_query_affinity_score'])
test['exponential_query_affinity_score'] = np.exp(test['srch_query_affinity_score'])



### drop columns
# Drop the comp_rate_percent_diff columns and separate rate and inv cols
train = train.drop(columns=comp_rate_percent_diff_cols)
test = test.drop(columns=comp_rate_percent_diff_cols)
train = train.drop(columns=comp_rate_cols + comp_inv_cols)
test = test.drop(columns=comp_rate_cols + comp_inv_cols)

# drop srch_adults_count and srch_children_count, because used in feature
train.drop(['srch_adults_count', 'srch_children_count'], axis=1, inplace=True)
test.drop(['srch_adults_count', 'srch_children_count'], axis=1, inplace=True)

# drop gross_bookings_usd, because it is not available in the test set
train.drop('gross_bookings_usd', axis=1, inplace=True)

