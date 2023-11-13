import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def handle_null_values(df):
    # Fill null values with the mean of the column
    df.fillna(df.mean(), inplace=True)
    return df

def handle_outliers(df, column):
    # Assuming 'df' is your DataFrame and 'column' is the column to check for outliers
    sns.boxplot(x=df[column])

    # Identify outliers
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Decide what to do with outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


def discretize_column(df, column):
    # Assuming 'df' is your DataFrame and 'column' is the column to discretize
    bins = [0, 25, 50, 75, 100]  # define your bins
    labels = ['Group1', 'Group2', 'Group3', 'Group4']  # define labels for each bin
    df['discrete_column'] = pd.cut(df[column], bins=bins, labels=labels)
    return df


def normalize_column(df, column):
    # Assuming 'df' is your DataFrame and 'column' is the column to normalize
    scaler = MinMaxScaler()
    df['normalized_column'] = scaler.fit_transform(df[[column]])
    return df

if __name__ == '__main__':
    columns = ['title', 'text', 'subject', 'date']

    # Read the CSV file
    True_df = pd.read_csv('True.csv')
    Fake_df = pd.read_csv('Fake.csv')

    # Fill null values with the mean of the column
    True_df = handle_null_values(True_df)
    Fake_df = handle_null_values(Fake_df)

    True_outliers = []
    Fake_outliers = []
    True_desicretized_df = pd.DataFrame()
    Fake_desicretized_df = pd.DataFrame()
    True_normalized_df = pd.DataFrame()
    Fake_normalized_df = pd.DataFrame()

    for column in columns:
        # Decide what to do with outliers
        True_outliers += handle_outliers(True_df, column)
        Fake_outliers += handle_outliers(Fake_df, column)
        # desicretize column
        True_desicretized_df += discretize_column(True_df, column)
        Fake_desicretized_df += discretize_column(Fake_df, column)
        # normalize column
        True_normalized_df += normalize_column(True_df, column)
        Fake_normalized_df += normalize_column(Fake_df, column)


