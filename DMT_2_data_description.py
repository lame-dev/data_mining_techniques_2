import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/training_set_VU_DM.csv')

def hist_plot(variable):
    plt.figure()
    df[variable].hist(bins=50, color='orange', edgecolor='black')
    plt.title(f'Histogram of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.grid(True)  # Add grid
    plt.show()
    #save the plot
    plt.savefig(f'plots/histogram_{variable}.png')
def plot_corr_matrix(corr_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()
    plt.savefig(f'plots/{title}.png')
def scatter_plot(column1, column2):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=column1, y=column2, data=df, marker='o', color='blue')
    plt.title(column1 + ' vs ' + column2)
    plt.show()
    plt.savefig(f'plots/scatter_{column1}_{column2}.png')

###################################################### Data ###########################################################
### Data description ###
### 1.1
# print shape of df
print(df.shape)
print(df.isnull().sum()) # print number of missing values in each column
print(df.isnull().mean() * 100) # print percentage of missing values in each column
print(df.columns[df.isnull().mean() == 0].shape[0]) # count number of columns that are full
print(df['click_bool'].value_counts()) #count number of clicks
print(df['booking_bool'].value_counts()) #count number of bookings
print(df['random_bool'].value_counts()) #count number of uniques in random bool

### plots for 1.2

#histograms:
#hist_plot('visitor_hist_starrating')
#hist_plot('visitor_hist_adr_usd')
#hist_plot('orig_destination_distance')
#hist_plot('prop_starrating')
#hist_plot('prop_review_score')
#hist_plot('prop_location_score1')
#hist_plot('prop_log_historical_price')
#hist_plot('srch_length_of_stay')
#hist_plot('srch_adults_count')
#could also be interesting: srch_query_affinity_score, srch_children_count, srch_room_count



# List of search criteria variables
heatmap_search_criteria = ['srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count',
                           'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score']
heatmap_hotel_criteria = ['prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1',
                          'prop_location_score2', 'prop_log_historical_price',  'promotion_flag']  # ,'price_usd'

corr_matrix_search = df[heatmap_search_criteria].corr()  # Create a correlation matrix for search criteria
corr_matrix_hotel = df[heatmap_hotel_criteria].corr()  # Create a correlation matrix for hotel criteria

plot_corr_matrix(corr_matrix_search, 'Search Criteria Correlation Matrix')
plot_corr_matrix(corr_matrix_hotel, 'Hotel Criteria Correlation Matrix')

#scatterplot, wat interessant???
#scatter_plot('prop_starrating', 'prop_log_historical_price')


### 1.3 barplot for missing values
# Calculate the percentage of complete values for each column
def plot_missing_values(df):
    # Calculate the percentage of missing values for each column
    missing_values_percent = df.isnull().mean() * 100
    missing_values_df = pd.DataFrame({'column_name': df.columns,
                                      'percent_missing': missing_values_percent})
    missing_values_df.sort_values('percent_missing', inplace=True, ascending=False)

    # Create a bar plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='percent_missing', y='column_name', data=missing_values_df, color = 'orange')
    plt.title('Percentage of missing values per variable')
    plt.xlabel('Percentage of missing values')
    plt.ylabel('Variable')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 101, 10), minor=True)  # Set minor ticks at every 10 units on x-axis
    ax.grid(which='both', axis='x', linestyle='--')  # Draw grid lines at minor ticks
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.invert_xaxis()
    plt.show()
    plt.savefig('plots/missing_values_barplot.png')

plot_missing_values(df)

def plot_missing_values_only_missing(df):
    # Calculate the percentage of missing values for each column
    missing_values_percent = df.isnull().mean() * 100
    # Filter out the columns that have no missing values
    missing_values_percent = missing_values_percent[missing_values_percent > 0]
    missing_values_df = pd.DataFrame({'column_name': missing_values_percent.index,
                                      'percent_missing': missing_values_percent.values})
    missing_values_df.sort_values('percent_missing', inplace=True, ascending=False)

    # Create a bar plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='percent_missing', y='column_name', data=missing_values_df, color='orange')
    plt.title('Percentage of missing values per variable')
    plt.xlabel('Percentage of missing values')
    plt.ylabel('Variable')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 101, 10), minor=True)  # Set minor ticks at every 10 units on x-axis
    ax.grid(which='both', axis='x', linestyle='--')  # Draw grid lines at minor ticks
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.invert_xaxis()
    plt.show()
    plt.savefig('plots/missing_values_barplot_only_missing.png')

plot_missing_values_only_missing(df)
# find outliers with boxplots
def boxplot(variable):
    plt.figure()
    sns.boxplot(x=df[variable], color='orange')
    plt.title(f'Boxplot of {variable}')
    plt.xlabel(variable)
    plt.show()
    plt.savefig(f'plots/boxplot_{variable}.png')


def plot_boxplots(df):
    # Select only the 'comp' columns
    comp_cols = [col for col in df.columns if 'comp' in col]
    df_comp = df[comp_cols]

    # Melt the DataFrame to make it suitable for boxplots
    df_melted = df_comp.melt(var_name='variable', value_name='value')

    # Create a boxplot for each variable
    plt.figure(figsize=(12, 6))  # Adjust the size as necessary
    sns.boxplot(x='variable', y='value', data=df_melted)
    plt.title('Boxplots for comp variables')
    plt.xticks(rotation=45)  # Rotates the labels on the x-axis to make them more readable
    plt.show()

# Call the function with your DataFrame
plot_boxplots(df)









#######################################################################################################################
### corr matrices
# correlation matrix search criteria, without desination id
heatmap_search_criteria = ['srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count',
                           'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score']
# correlation matrix star, review, brand, loc score 1, loc score 2, hist price, price, promotion flag
heatmap_hotel_criteria = ['prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1',
                          'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag']

corr_matrix_search = df[heatmap_search_criteria].corr()  # Create a correlation matrix for search criteria
corr_matrix_hotel = df[heatmap_hotel_criteria].corr()  # Create a correlation matrix for hotel criteria
plot_corr_matrix(corr_matrix_search, 'Search Criteria Correlation Matrix')
plot_corr_matrix(corr_matrix_hotel, 'Hotel Criteria Correlation Matrix')


### competitor rate plot
# Filter for competitor rate columns and the necessary target columns
# Filter for competitor rate columns and the necessary target columns
comp_cols = [col for col in df.columns if 'comp' in col and 'rate' in col and 'percent' not in col]
df_comp = df[comp_cols + ['booking_bool', 'click_bool']]

# Calculate booking and click percentages
results = []

for col in comp_cols:
    for value in [-1, 0, 1]:  # -1 for Lower, 0 for Match, 1 for Better
        total = len(df_comp[df_comp[col] == value])
        bookings = df_comp[(df_comp[col] == value) & (df_comp['booking_bool'] == 1)].shape[0]
        clicks = df_comp[(df_comp[col] == value) & (df_comp['click_bool'] == 1)].shape[0]
        if total > 0:
            booking_percentage = (bookings / total) * 100
            click_percentage = (clicks / total) * 100
        else:
            booking_percentage = 0
            click_percentage = 0
        results.append([col, value, 'Booking', booking_percentage])
        results.append([col, value, 'Click', click_percentage])

# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['Competitor Rate Comparison', 'Value', 'Type', 'Percentage'])
value_mapping = {-1: '>', 0: '=', 1: '<'}
results_df['Value'] = results_df['Value'].map(value_mapping)
results_df['Competitor Rate Comparison'] = results_df['Competitor Rate Comparison'].str.replace('_rate', '') + '_' + results_df['Value']

# Insert empty labels for spacing
unique_comps = results_df['Competitor Rate Comparison'].str.extract(r'(comp\d+)_')[0].unique()
dummy_rows = []

for comp in unique_comps:
    for val in ['<', '=', '>']:
        dummy_rows.append({'Competitor Rate Comparison': f'{comp}_dummy', 'Value': '', 'Type': '', 'Percentage': 0})

# Convert the list of dummy rows to a DataFrame
dummy_df = pd.DataFrame(dummy_rows)

# Concatenate the dummy rows with the original DataFrame
results_df = pd.concat([results_df, dummy_df], ignore_index=True)

# Sort the DataFrame to maintain order
results_df['sort_key'] = results_df['Competitor Rate Comparison'].str.extract(r'(comp\d+)_(.*)')[0] + results_df['Value']
results_df = results_df.sort_values(by='sort_key').drop(columns='sort_key')

# Plotting
plt.figure(figsize=(14, 8))
sns.barplot(x='Competitor Rate Comparison', y='Percentage', hue='Type', data=results_df, dodge=True)
plt.xlabel('Rate Comparison')
plt.ylabel('% Bookings / Clicks')
plt.title('Percentage of Bookings and Clicks by Competitor Rate Comparison')
plt.xticks(rotation=90)

# Remove the dummy labels from x-axis
current_labels = plt.gca().get_xticklabels()
new_labels = ['' if 'dummy' in label.get_text() else label.get_text() for label in current_labels]
plt.gca().set_xticklabels(new_labels)

plt.legend(title='')
plt.tight_layout()
plt.show()
plt.savefig('plots/comp_rate_plot.png')






### position bias plot
# Summarize data by position and random sorting
summary_df = df.groupby(['position', 'random_bool']).agg(
    booking_count=('booking_bool', 'sum'),
    click_count=('click_bool', 'sum')
).reset_index()

# Melt the DataFrame for plotting
summary_melted = summary_df.melt(
    id_vars=['position', 'random_bool'],
    value_vars=['booking_count', 'click_count'],
    var_name='type',
    value_name='count'
)

# Map random_bool to human-readable labels
summary_melted['random_bool'] = summary_melted['random_bool'].map({0: 'non-random', 1: 'random'})
summary_melted['type'] = summary_melted['type'].map({'booking_count': 'booking', 'click_count': 'click'})

# Create the FacetGrid
g = sns.FacetGrid(summary_melted, col='random_bool', height=6, aspect=2)

# Map the barplot onto the FacetGrid
g.map_dataframe(sns.barplot, x='position', y='count', hue='type', palette=['#FF9999', '#66B2FF'])

# Adjust the plot
g.set_axis_labels("position", "# bookings / clicks")
g.set_titles(col_template="{col_name}")
g.add_legend(title='type')

# Show plot
plt.show()
plt.savefig('plots/position_bias_plot.png')

### purchase history is informative plot
# Calculate star_diff and price_diff
df['star_diff'] = df['visitor_hist_starrating'] - df['prop_starrating']
df['price_diff'] = df['visitor_hist_adr_usd'] - df['price_usd']

# Filter out rows where visitor history is not available
filtered_df = df.dropna(subset=['visitor_hist_starrating', 'visitor_hist_adr_usd'])

# Create a new column for user action categories
conditions = [
    (filtered_df['booking_bool'] == 1),
    (filtered_df['click_bool'] == 1) & (filtered_df['booking_bool'] == 0),
    (filtered_df['click_bool'] == 0) & (filtered_df['booking_bool'] == 0)
]
choices = ['booking', 'click', 'no click, no booking']
filtered_df['user_action'] = np.select(conditions, choices, default='unknown')

# Create the plot
plt.figure(figsize=(10, 6))

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

sns.barplot(x='user_action', y='star_diff', data=filtered_df, ax=axs[0])
axs[0].set_title('star_diff')

sns.barplot(x='user_action', y='price_diff', data=filtered_df, ax=axs[1])
axs[1].set_title('price_diff')

plt.suptitle('Purchase history is informative', fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('purchase_history_informative.png')

plt.show()


### other missing values: booking percentage nan against not nan for srch query affinity score, orig destination distance, visitor hist starrating, visitor hist adr usd, prop location score 2, prop review score
# Define the columns of interest
columns_of_interest = ['srch_query_affinity_score', 'orig_destination_distance', 'visitor_hist_starrating',
                       'visitor_hist_adr_usd', 'prop_location_score2', 'prop_review_score']

# Initialize lists to store percentages
bookings_nan = []
bookings_not_nan = []
clicks_nan = []
clicks_not_nan = []

# Calculate percentages for each column
for column in columns_of_interest:
    # Bookings
    booking_nan = df[df[column].isna()]['booking_bool'].mean() * 100
    booking_not_nan = df[~df[column].isna()]['booking_bool'].mean() * 100
    bookings_nan.append(booking_nan)
    bookings_not_nan.append(booking_not_nan)

    # Clicks
    click_nan = df[df[column].isna()]['click_bool'].mean() * 100
    click_not_nan = df[~df[column].isna()]['click_bool'].mean() * 100
    clicks_nan.append(click_nan)
    clicks_not_nan.append(click_not_nan)

# Create DataFrames for plotting
bookings_df = pd.DataFrame({
    'Feature': columns_of_interest,
    'NaN': bookings_nan,
    'Not NaN': bookings_not_nan
}).melt(id_vars='Feature', var_name='Type', value_name='Percentage')

clicks_df = pd.DataFrame({
    'Feature': columns_of_interest,
    'NaN': clicks_nan,
    'Not NaN': clicks_not_nan
}).melt(id_vars='Feature', var_name='Type', value_name='Percentage')

# Create the plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Bookings plot
sns.barplot(x='Feature', y='Percentage', hue='Type', data=bookings_df, ax=axes[0])
axes[0].set_title('Bookings')
axes[0].set_ylabel('Percentage of hotels being booked')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=45)

# Clicks plot
sns.barplot(x='Feature', y='Percentage', hue='Type', data=clicks_df, ax=axes[1])
axes[1].set_title('Clicks')
axes[1].set_ylabel('Percentage of hotels being clicked')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)

# Set overall title and adjust layout
plt.suptitle('')
plt.tight_layout()
plt.show()
plt.savefig('nan_vs_not_nan_bookings_clicks.png')





### outlierplots: query affinity score, orig destination, visitor_hist_adr_usd, srch_length_of_stay, prop_log_historical_price, srch_booking_window, price_usd
# make top to bottom

# Define the columns to be plotted
# Define the columns to be plotted and sort them alphabetically
columns_to_plot = sorted(['visitor_hist_adr_usd', 'srch_length_of_stay', 'prop_log_historical_price',
                          'srch_booking_window', 'price_usd', 'srch_query_affinity_score', 'orig_destination_distance'])

# Create the boxplots
plt.figure(figsize=(20, 8))  # Increase the height for better visibility
df[columns_to_plot].plot(kind='box', subplots=True, layout=(1, len(columns_to_plot)),
                         sharey=False, vert=True, figsize=(20, 8), patch_artist=True)

# Set overall title and layout
plt.suptitle('Boxplots of variables with potential outliers', y=0.95)  # Adjust y for better visibility
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add rect to prevent title from getting cut off

# Show and save the plot
plt.show()
plt.savefig('boxplots_with_additional_variables.png')



### same as above for comp variables


# nog doen???