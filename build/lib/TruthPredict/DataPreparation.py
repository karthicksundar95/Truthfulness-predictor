### Data preprocessing and encoding
'''
    File name: DataPreparation.py
    Author: Karthick Sundar C K
    Date created: 01/02/2024
    Date last modified: 07/02/2024
    Python Version: 3.10
'''
# Importing data handling python packages
import pandas as pd
import re
import string
from scipy.sparse import csr_matrix
import scipy as sp
import contractions
import joblib
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,  MultiLabelBinarizer


from TruthPredict.constants import *
stopwords_list = stopwords.words('english')


class DataPrep:
    """
    Class module to do data pre-processing, encoding and formatting on the incoming train or val or test data
    """
    def __init__(self):
        """
        Class constructor which initializes essential variable for the class methods
        """
        self.data = pd.DataFrame()
        self.stopwords_list = stopwords.words('english')
        self.lemma = WordNetLemmatizer()
        self.tf_idf_train_vector = {}
        self.tf_idf_test_vector = {}
        self.X_train_column_names = []

    def prep(self, run, model_save_path):
        """
        Main method that controls the flow of pre-processing to be executed for train and test condition checks
        :param run: mode of run, either "train" or "test"
        :return:
        """
        self.run = run
        self.model_save_path = model_save_path
        if self.run == 'train':
            # target label normalization is only executed in case of running it in "train" mode only
            self.normalize_target_label()
            print(" >>>   \nTarget label data is normalized to binary TRUE/FALSE  >>>")
        self.multi_subject_encoding()
        print(" >>>   Multi-labeled subject column is converted to encoded format   >>>")
        self.one_hot_encoding('speaker_name')  # No N/As
        print(" >>>   Speaker column is converted to encoded format   >>>")
        self.one_hot_encoding('speaker_job')
        print(" >>>   Speaker_job column is converted to encoded format   >>>")
        self.one_hot_encoding('speaker_state')
        print(" >>>   Speaker_state column is converted to encoded format   >>>")
        self.one_hot_encoding('speaker_affiliation')
        print(" >>>   Speaker_affiliation column is converted to encoded format   >>>")
        self.preprocess_text([STATEMENT,STATEMENT_CONTEXT])
        print(""" >>>   Statement column processed as follows, \n
                      1. Punctuations removed\n
                      2. Stopwords removed\n
                      3. Contractions expanded\n
                      4. Lemmatization >>> """)

    def read_data(self, filepath):
        """
        Method to read the csv file with data for training
        :param filepath: path where the file is located
        :return: dataframe with data read from csv file
        """
        self.data = pd.read_csv(filepath)

    def normalize_target_label(self):
        """
        Method to normalize the 6 level target to bonary label format TRUE/FALSE
        :return: dataframe target column with only TRUE/FALSE labels
        """
        target_labels = set(self.data[TARGET_COLUMN].unique())
        if target_labels.issubset(['true', 'false']):
            pass
        else:
            self.data.loc[self.data[TARGET_COLUMN].isin(['mostly-true', 'half-true', 'true']), TARGET_COLUMN] = 'TRUE'
            self.data.loc[
                self.data[TARGET_COLUMN].isin(['false', 'extremely-false', 'barely-true']), TARGET_COLUMN] = 'FALSE'

    def multi_subject_encoding(self):
        """
        Encoding of multilabeled column "subjects" using multilabelBinarizer and saving the model as pkl file.
        Model is loaded to transform the test input when run using "test"(inference) mode
        :return: pickle file with encoder model
        """
        if self.run == 'train':
            mlb = MultiLabelBinarizer()
            encoded_df = pd.DataFrame(mlb.fit_transform(self.data[SUBJECTS].apply(lambda x: [j.strip() for j in x.split("$")])),
                                      columns=mlb.classes_,index=self.data[SUBJECTS].index)
            joblib.dump(mlb, Path(self.model_save_path).joinpath("subjects_encoder.pkl"))
            self.data = pd.merge(self.data.drop([SUBJECTS], axis=1), encoded_df, left_index=True, right_index=True)
        elif self.run == 'test':
            mlb = joblib.load( Path(self.model_save_path).joinpath("subjects_encoder.pkl"))
            encoded_df = pd.DataFrame(mlb.transform(self.data[SUBJECTS].apply(lambda x: [j.strip() for j in x.split("$")])),
                                     columns=mlb.classes_,index=self.data[SUBJECTS].index)
            self.data = pd.merge(self.data.drop([SUBJECTS], axis=1), encoded_df, left_index=True, right_index=True)

    def one_hot_encoding(self, column_name):
        """
        Method to one-hot-encode any categorical column input
        :param column_name: name of the column with categorical data with ordinal feature
        :return: dataframe with the given categorical columns one-hot encoded
        """
        # removing null values and cleaning the text
        self.data[column_name] = self.data[column_name].apply(
            lambda x: "" if str(x).lower() in ['none', 'unknown', 'n/a', ""]
            else re.sub("\s+", " ", re.sub('[^a-zA-Z]', ' ',
                                           re.sub("\.", "", str(x).lower().strip()).strip())))
        if self.run == 'train':
            # creating one-hot-encoding and saving the model
            encoder = OneHotEncoder()
            one_hot_array = encoder.fit_transform(self.data[[column_name]]).toarray()
            joblib.dump(encoder,  Path(self.model_save_path).joinpath(column_name + '_' + 'vectorizer.pkl'))
        else:
            # loading the trained model to transform the new test data for inference
            encoder = joblib.load( Path(self.model_save_path).joinpath(column_name + '_' + 'vectorizer.pkl'))
            one_hot_array = encoder.transform(self.data[[column_name]]).toarray()

        # formatting the one-hot-encoded output with respective column names
        one_hot_df = pd.DataFrame(one_hot_array, columns=encoder.get_feature_names_out())
        self.data = pd.merge(self.data.drop([column_name], axis=1), one_hot_df, left_index=True, right_index=True)

        # dropping unnecessary columns from null(nan or blank) values from the encoding output
        try:
            self.data.drop([column_name + "_nan"], axis=1, inplace=True)  # removing nan encoded colum
        except:
            pass
        try:
            self.data.drop([column_name + "_"], axis=1, inplace=True)  # removing blank encoded colum
        except:
            pass

    def tokenization_punct(self, text):
        """
        To tokenize the text data into words and punctuations
        :param text: input text string
        :return: list of tokenized words and punctuations
        """
        tokens = re.findall(r"[\w]+|[^\s\w]", str(text))
        return tokens

    def remove_punctuation(self, text):
        """
        To remove all punctuations known in English language
        :param text: input test string
        :return: list of tokens with punctuations removed
        """
        punctuationfree = [i for i in text if i not in string.punctuation]
        return punctuationfree

    def remove_stopwords(self, text):
        """
        To remove commonly used words in constructing an english sentence grammatically correct
        :param text: input test string
        :return: list of words which only convey a meaning
        """
        output = [i for i in text if i not in self.stopwords_list]
        return output

    def lemmatizer(self, text):
        """
        To convert each word into its root form
        :param text: input test string
        :return: string made by combining list of lemmatized words
        """
        lemm_text = [self.lemma.lemmatize(word, pos='v').lower() for word in text]
        return " ".join(lemm_text)

    def remove_numbers(self, text):
        """
        To remove any numbers in the test data
        :param text: input test string
        :return: test with all numbers removed
        """
        result = re.sub(r'\d+', '', text)
        return result

    def contractions_fix(self, text):
        """
        Method to expand all the contracted words
        :param text: input text string
        :return: list of words with expanded format of contractions if any
        """
        expanded_text = [contractions.fix(word) for word in text]
        return expanded_text

    def preprocess_text(self,text_columns):
        """
        Method to execute sequence of actions from tokenizing, stopword and number removal,
        contractions expansion and root word conversion(lemmatization)
        :param text_columns: input text string
        :return: string with all cleaning and formatting
        """
        for column in text_columns:
            self.data[column] = self.data[column].apply(lambda x: self.tokenization_punct(x))
            self.data[column] = self.data[column].apply(lambda x: self.remove_punctuation(x))
            self.data[column] = self.data[column].apply(lambda x: self.remove_stopwords(x))
            self.data[column] = self.data[column].apply(lambda x: self.contractions_fix(x))
            self.data[column] = self.data[column].apply(lambda x: self.lemmatizer(x))
            self.data[column] = self.data[column].apply(lambda x: self.remove_numbers(x))

    def train_test_split(self):
        """
        Method to split the data to perfrom training on 67% of data and validation of model
        performance on 33% data
        :return: pandas dataframe of train and test split
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.loc[:, self.data.columns != TARGET_COLUMN], self.data[TARGET_COLUMN],
            test_size=0.33, random_state=42, stratify=self.data[TARGET_COLUMN])

    def tfidf_vectorizer(self, column_name):
        """
        To convert textual columns into vector fromat using TFIDF method
        :param column_name: text column name
        :return: scipy matrix of vectors
        """
        if self.run == 'train':
            self.tf_idf_model = TfidfVectorizer(min_df=5)
            self.tf_idf_train_vector[column_name] = self.tf_idf_model.fit_transform(
                self.X_train[column_name].values.astype('str'))
            self.tf_idf_test_vector[column_name] = self.tf_idf_model.transform(
                self.X_test[column_name].values.astype('str'))
            joblib.dump(self.tf_idf_model,  Path(self.model_save_path).joinpath(column_name + '_' + 'vectorizer.pkl'))
            self.X_train = self.X_train.drop(column_name, axis=1)
            self.X_test = self.X_test.drop(column_name, axis=1)
            self.X_train_column_names.extend(self.tf_idf_model.get_feature_names_out())
        else:
            self.tf_idf_model = joblib.load(Path(self.model_save_path).joinpath(column_name + '_' + 'vectorizer.pkl'))
            self.X_train_column_names.extend(self.tf_idf_model.get_feature_names_out())
            self.tf_idf_test_vector[column_name] = self.tf_idf_model.transform(self.data[column_name].values.astype('str'))
            self.data = self.data.drop(column_name, axis=1)

    def merge_features(self, text_columns):
        """
        Method to merge all the text vectors and the one-hot-encoded categorical vectors into single csr matrix
        :param text_columns: columns to be included to merge dataframes
        :return: csr matrix
        """
        if self.run == 'train':
            self.X_train_column_names.extend(self.X_train.columns)
            for column in text_columns:
                self.X_train = sp.sparse.hstack((self.tf_idf_train_vector[column], csr_matrix(self.X_train)))
                self.X_test = sp.sparse.hstack((self.tf_idf_test_vector[column], csr_matrix(self.X_test)))
            print("\nFinal feature size:", self.X_train.shape, self.X_test.shape)
        else:
            self.X_train_column_names.append(self.data.columns)
            for column in text_columns:
                self.data = sp.sparse.hstack((self.tf_idf_test_vector[column], csr_matrix(self.data)))




