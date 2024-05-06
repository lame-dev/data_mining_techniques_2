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
