import math

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import datetime as dt
import textacy
import textacy.preprocessing as tp
from textacy import text_stats
from textacy import extract
import spacy


def handle_missing_values(df):
    for i in range(len(df['text'])):
        if df['text'][i] == np.nan:
            df.drop(i, inplace=True)
        if df['title'][i] == np.nan:
            df.drop(i, inplace=True)


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
    #df['date'] = pd.to_datetime(df['date'])
    for i in range(len(df['date'])):
        #print(df['date'][i])
        df['date'][i] = pd.to_datetime(df['date'][i])
        #print(df['date'][i])


    return df


# Date is the only column that can be desicretized
def discretize_date(df):
    labels = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    for i in range(len(df['date'])):
        df['descrete_date'][i] = pd.cut(df['date'][i].dt.month, 12, labels=labels)
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
        print(i)
        try:
            df['text'][i] = tp.normalize.whitespace(df['text'][i])
            df['text'][i] = tp.normalize.hyphenated_words(df['text'][i])
            df['text'][i] = tp.normalize.quotation_marks(df['text'][i])
            df['text'][i] = tp.normalize.unicode(df['text'][i])
            df['text'][i] = tp.remove.accents(df['text'][i])
            df['text'][i] = tp.remove.punctuation(df['text'][i])
            df['text'][i] = tp.remove.punctuation(df['text'][i])
        except:
            #print(df['text'][i])
            #df.drop(index=i, inplace=True)
            print("error")

        try:
            df['title'][i] = tp.normalize.whitespace(df['title'][i])
            df['title'][i] = tp.normalize.hyphenated_words(df['title'][i])
            df['title'][i] = tp.normalize.quotation_marks(df['title'][i])
            df['title'][i] = tp.normalize.unicode(df['title'][i])
            df['title'][i] = tp.remove.accents(df['title'][i])
            df['title'][i] = tp.remove.punctuation(df['title'][i])
            df['title'][i] = tp.remove.punctuation(df['title'][i])
        except:
            #print(df['title'][i])
            #df.drop(index=i, inplace=True)
            print("error")

    return df


# feature extraction with sklearn
def feature_extraction_text(df):
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(df['text'])
    return counts


# feature extraction with textacy
def feature_extraction_textacy(df):
    npl = spacy.load('en_core_web_sm')
    sample = df['text'][0]
    doc = npl(sample)
    print("number of sentences")
    print(text_stats.basics.n_sents(doc))
    print("number of words")
    print(text_stats.basics.n_words(doc))
    print("number of unique words")
    print(text_stats.basics.n_unique_words(doc))
    print("lexical density")
    print(text_stats.diversity.ttr(doc))
    print("readability by grade level")
    print(text_stats.readability.flesch_kincaid_grade_level(doc))

    en3 = extract.basics.ngrams(doc, n=3, filter_stops=True, filter_punct=True, filter_nums=False, include_pos=None, exclude_pos=None, min_freq=3)
    print(list(en3))

    print(extract.keyword_in_context(doc))

    em = extract.matches.token_matches(doc, [r'(?i)obama', r'(?i)trump'])
    print(list(em))

if __name__ == '__main__':
    columns = ['title', 'text', 'subject', 'date']

    # Read the CSV file
    True_f = pd.read_csv('True_modified2.csv')
    Fake_f = pd.read_csv('Fake_modified2.csv')
    print("read csv files")

    # Create a DataFrame
    True_df = pd.DataFrame(True_f)
    Fake_df = pd.DataFrame(Fake_f)

    # Handle missing values
    #handle_missing_values(True_df)
    #handle_missing_values(Fake_df)

    # Convert date to datetime
    #True_df = convert_date(True_df)
    #Fake_df = convert_date(Fake_df)
    #print("converted date format")
    #counts = pd.Series(index=True_df['date'], data=np.array(True_df.count)).resample('M').count()
    #print(counts)

    # Discretize date
    #True_df = discretize_date(True_df)
    #Fake_df = discretize_date(Fake_df)
    #print("desicretized date")
    #print(True_df)

    # Normalize text
    #True_df = normalize_text(True_df)
    #Fake_df = normalize_text(Fake_df)
    #print("normalized text")

    # Export to csv
    #export_csv(True_df, "True_modified2")
    #export_csv(Fake_df, "Fake_modified2")

    # Feature extraction
    feature_extraction_textacy(True_df)


    # Export to csv
    # export_csv(True_df, "True_modified")
    # export_csv(Fake_df, "Fake_modified")

