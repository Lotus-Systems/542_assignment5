import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import textacy
import textacy.preprocessing as tp


def handle_outliers(df, column):
    sns.boxplot(y=df[column])

    # Identify outliers
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Decide what to do with outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


def convert_date(df):
    df['date'] = pd.to_datetime(df['date'])
    #for i in range(len(df['date'])):
        #print(df['date'][i])
        #df['date'][i] = pd.to_datetime(df['date'][i])
        #print(df['date'][i])

    return df


# Date is the only column that can be desicretized
def discretize_date(df):
    labels = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    df['discrete_date'] = pd.cut(df['date'].dt.month, 12, labels=labels)
    return df


def normalize_column(df, column):
    scaler = MinMaxScaler()
    df['normalized_column'] = scaler.fit_transform(df[[column]])
    return df


# export dataframe to csv
def export_csv(df, name):
    df.to_csv(name + '.csv', index=False)


# normalize text and titles
def normalize_text(df):
    for i in range(len(df['text'])):
        df['text'][i] = tp.normalize.normalize_whitespace(df['text'][i])
        df['text'][i] = tp.normalize.normalize_hyphenated_words(df['text'][i])
        df['text'][i] = tp.normalize.normalize_quotation_marks(df['text'][i])
        df['text'][i] = tp.normalize.normalize_unicode(df['text'][i])
        df['text'][i] = tp.normalize.normalize_whitespace(df['text'][i])
        df['text'][i] = tp.normalize.remove_accents(df['text'][i])
        df['text'][i] = tp.normalize.remove_punctuation(df['text'][i])
        df['text'][i] = tp.normalize.remove_stopwords(df['text'][i])
        df['text'][i] = tp.normalize.remove_punctuation(df['text'][i])

    for i in range(len(df['title'])):
        df['title'][i] = tp.normalize.normalize_whitespace(df['title'][i])
        df['title'][i] = tp.normalize.normalize_hyphenated_words(df['title'][i])
        df['title'][i] = tp.normalize.normalize_quotation_marks(df['title'][i])
        df['title'][i] = tp.normalize.normalize_unicode(df['title'][i])
        df['title'][i] = tp.normalize.normalize_whitespace(df['title'][i])
        df['title'][i] = tp.normalize.remove_accents(df['title'][i])
        df['title'][i] = tp.normalize.remove_punctuation(df['title'][i])
        df['title'][i] = tp.normalize.remove_stopwords(df['title'][i])
        df['title'][i] = tp.normalize.remove_punctuation(df['title'][i])

    return df


if __name__ == '__main__':
    columns = ['title', 'text', 'subject', 'date']

    # Read the CSV file
    True_f = pd.read_csv('True.csv')
    Fake_f = pd.read_csv('Fake.csv')
    print("read csv files")

    # Create a DataFrame
    True_df = pd.DataFrame(True_f)
    Fake_df = pd.DataFrame(Fake_f)

    # Convert date to datetime
    True_df = convert_date(True_df)
    Fake_df = convert_date(Fake_df)
    print("converted date format")
    #counts = pd.Series(index=True_df['date'], data=np.array(True_df.count)).resample('M').count()
    #print(counts)

    # Discretize date
    True_df = discretize_date(True_df)
    Fake_df = discretize_date(Fake_df)
    #print("desicretized date")
    #print(True_df)

    # Normalize text
    True_df = normalize_text(True_df)
    Fake_df = normalize_text(Fake_df)
    print("normalized text")

    # Export to csv
    # export_csv(True_df, "True_modified")
    # export_csv(Fake_df, "Fake_modified")

