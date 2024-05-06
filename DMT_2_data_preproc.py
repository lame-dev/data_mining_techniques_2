import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data MOET DIT AL DF_TRAIN ZIJN??????? ZET GOED IN REPORT WELKE HET IS
df = pd.read_csv('data/training_set_VU_DM.csv')

def print_missing_values_columns(df): #check in between steps what columns are left to process
    missing_values_percent = df.isnull().mean() * 100
    missing_values_columns = missing_values_percent[missing_values_percent > 0]
    print(missing_values_columns)

#print_missing_values_columns(df) #show all columns with missing values and percentages


# Impute comp columns missing values with zero (following slide 75)
comp_rate_cols = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']
comp_inv_cols = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv']
comp_rate_percent_diff_cols = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff','comp7_rate_percent_diff', 'comp8_rate_percent_diff']

comp_cols = comp_rate_cols + comp_inv_cols + comp_rate_percent_diff_cols
df[comp_cols] = df[comp_cols].fillna(0)
print_missing_values_columns(df)

# leftover missing value columns
columns_to_plot_barzzzz = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_review_score',
                   'prop_location_score2', 'srch_query_affinity_score', 'orig_destination_distance',
                   'gross_bookings_usd']

## Loop over the columns and create a plot for each
#for column in columns_to_plot_barzzzz:
#    plt.figure(figsize=(10, 6))
#    # Create a new column that indicates whether the current column is NA or not NA
#    df[column + '_isnull'] = df[column].isnull()
#
#    # Group by the new column and calculate the mean of 'booking_bool'
#    booking_percent = df.groupby(column + '_isnull')['booking_bool'].mean() * 100
#
#    # Plot the result as a histogram
#    booking_percent.plot(kind='bar', color='blue')
#    plt.xlabel(f'{column} is NA')
#    plt.ylabel('Percentage of hotels booked (%)')
#    plt.title(f'Percentage of hotels booked vs. NA/not NA {column}')
#    plt.xticks([0, 1], ['Not NA', 'NA'], rotation=0)  # Replace the x-axis labels
#    plt.show()
#    plt.savefig(f'plots/percentage_booked_vs_{column}.png')

##### FILLING MISSING VALUES
# fill the missing values for orig_destination_distance with 0
df['orig_destination_distance'] = df['orig_destination_distance'].fillna(0)

columns_to_fill = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_review_score',
                   'prop_location_score2', 'srch_query_affinity_score', 'gross_bookings_usd']

# Calculate the first quantile for each column and fill missing values
for column in columns_to_fill:
    first_quantile = df[column].quantile(0.25)
    df[column] = df[column].fillna(first_quantile)

print_missing_values_columns(df) #check that all columns now dont have any missing values anymore


######### outliers
# plots
columns_to_plot = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_review_score',
                   'prop_location_score2', 'srch_query_affinity_score', 'orig_destination_distance',
                   'gross_bookings_usd']

# Filling the missing values for orig_destination_distance with 0
df['orig_destination_distance'] = df['orig_destination_distance'].fillna(0)

#for column in df.columns:
#    # Create a figure for the boxplot
#    plt.figure(figsize=(10, 6))
#    sns.boxplot(x=df[column])
#
#    # Adding labels and title
#    plt.title(f'Boxplot for {column}')
#    plt.xlabel(column)
#    plt.ylabel('Values')
#
#    # Save the plot in the 'outlierplots' directory
#    plt.savefig(f'outlierplots/boxplot_{column}.png')
#    plt.close()  # Close the plot to avoid display overhead and memory issues


######################################## feature engineering ########################################
#new features

#two new features because (slide 74)
df['starrating_diff'] = np.abs(df['visitor_hist_starrating'] - df['prop_starrating']) #feature 1
df['price_diff'] = np.abs(df['visitor_hist_adr_usd'] - df['price_usd']) #feature 2

# feature 3: booking_probability and feature 4: click_probability

# create prob_book and prob_click variables
grouped = df.groupby('prop_id').agg(
    num_bookings=pd.NamedAgg(column='booking_bool', aggfunc='sum'),  # Sum bookings to get total bookings
    num_clicks=pd.NamedAgg(column='click_bool', aggfunc='sum'),      # Sum clicks to get total clicks
    total_listings=pd.NamedAgg(column='prop_id', aggfunc='size')     # Count the number of listings
)

# Calculate booking and click probabilities
grouped['booking_probability'] = grouped['num_bookings'] / grouped['total_listings']
grouped['click_probability'] = grouped['num_clicks'] / grouped['total_listings']
grouped.reset_index(inplace=True)# Reset index to merge back
df = df.merge(grouped[['prop_id', 'booking_probability', 'click_probability']], on='prop_id', how='left')

# feature 5:






############################ do this with batching because takes a long time
### create averaged features
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
excluded_cols = {'srch_id', 'prop_id', 'srch_destination_id', # columns to average over
                 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_brand_bool', 'promotion_flag',
                 'srch_saturday_night_bool', 'random_bool', 'click_bool', 'booking_bool' + comp_rate_cols + comp_inv_cols}
                 # IDs and other non-relevant columns if any
numerical_cols3 = [col for col in numerical_cols if col not in excluded_cols]

def compute_and_merge(df, group_col, feature_cols):
    averages = df.groupby(group_col)[feature_cols].mean().reset_index()    # Compute the mean
    averages.rename(columns={col: f'{col}_avg_{group_col}' for col in feature_cols}, inplace=True)    # Rename the columns
    return df.merge(averages, on=group_col, how='left')    # Merge the averages back to the original DataFrame

# Apply the function for each group_col
df = compute_and_merge(df, 'srch_id', numerical_cols)
df = compute_and_merge(df, 'prop_id', numerical_cols)
df = compute_and_merge(df, 'srch_destination_id', numerical_cols)




####### ideas features
# gross bookings naar nieuwe feature
# include month feature or at least get rid of date time
# at the end, averagea over numerical variables
#
