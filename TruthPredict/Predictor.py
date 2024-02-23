### Truth Fullness predictor
'''
    File name: Predictor.py
    Author: Karthick Sundar C K
    Date created: 01/02/2024
    Date last modified: 07/02/2024
    Python Version: 3.10
'''
# Importing data handling python packages
from TruthPredict.DataPreparation import DataPrep
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import joblib
import json
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report

from TruthPredict.constants import *


class TruthClassifier(DataPrep):
    """
    Class module to predict truthfulness in the statements made by noteable peopel from USA
    """
    def __init__(self):
        """
        Class constructor to declare and initialize variables and parent class
        """
        DataPrep.__init__(self)
        self.k_folds = KFold(n_splits=CVSPLITS)
        self.model_metrics = {}

    def Naive_Bayes(self):
        """
        Method to build a Naive Bayes classifier
        :return: pickle model file
        """
        self.nb_model = MultinomialNB()
        nb_params = {'alpha': np.logspace(0, -9, num=50)}
        self.nb_model = GridSearchCV(self.nb_model, nb_params, cv=self.k_folds,
                                     refit=True, verbose=self.grid_search_verbose, scoring='accuracy')
        self.nb_model.fit(self.X_train, self.y_train)
        print("Best params: ", self.nb_model.best_params_)
        self.nb_model_best_params = self.nb_model.best_params_
        self.nb_model = MultinomialNB(**self.nb_model.best_params_).fit(self.X_train, self.y_train)
        self.predicted_categories = self.nb_model.predict(self.X_test)
        joblib.dump(self.nb_model, Path(self.model_save_path).joinpath('nb_model.pkl'))

    def Decision_Tree(self):
        """
        Method to build a decision tree classifier
        :return: pickle model file
        """
        self.dt_model = DecisionTreeClassifier()
        dt_params = {'max_depth': [2, 3, 5, 10, 20, 50],
                     'min_samples_leaf': [5, 10, 20, 50, 100],
                     # 'min_samples_split' : np.arange(2,11).tolist()[0::2],
                     #   'max_leaf_nodes' : np.arange(3,26).tolist()[0::2],
                     'criterion': ["gini", "entropy"]
                     }
        self.dt_model = GridSearchCV(self.dt_model, dt_params, cv=self.k_folds,
                                     refit=True, verbose=self.grid_search_verbose, scoring='accuracy')
        self.dt_model.fit(self.X_train, self.y_train)
        print("Best params: ", self.dt_model.best_params_)
        self.dt_model_best_params = self.dt_model.best_params_
        self.dt_model = DecisionTreeClassifier(**self.dt_model.best_params_).fit(self.X_train, self.y_train)
        self.predicted_categories = self.dt_model.predict(self.X_test)
        joblib.dump(self.dt_model, Path(self.model_save_path).joinpath('dt_model.pkl'))

    def Random_Forest(self):
        """
        Method to build a Random Forest classifier
        :return: pickle model file
        """
        self.rf_model = RandomForestClassifier(max_depth=20, n_estimators=200, random_state=42)
        rf_params = {'max_depth': [2, 3, 5],
                     'min_samples_leaf': [5, 10],
                     # 'min_samples_split' : np.arange(2,11).tolist()[0::2],
                     #   'max_leaf_nodes' : np.arange(3,26).tolist()[0::2],
                     'criterion': ["gini", "entropy"]
                     }
        self.rf_model = GridSearchCV(self.rf_model, rf_params, cv=self.k_folds,
                                     refit=True, verbose=self.grid_search_verbose, scoring='accuracy')
        self.rf_model.fit(self.X_train, self.y_train)
        print("Best params: ", self.rf_model.best_params_)
        self.rf_model_best_params = self.rf_model.best_params_
        self.rf_model = RandomForestClassifier(**self.rf_model.best_params_).fit(self.X_train, self.y_train)
        self.predicted_categories = self.rf_model.predict(self.X_test)
        joblib.dump(self.rf_model, Path(self.model_save_path).joinpath('rf_model.pkl'))

    def xgboost(self):
        """
        Method to build a random forest classifier
        :return: pickle model file
        """
        self.xgb_model = XGBClassifier(enable_categorical=True, max_cat_to_onehot=1)
        xgb_params = {'max_depth': [2, 3, 5 ,10],
                      #                       'max_depth': [2,3,5,10,20],
                      'learning_rate': [0.1, 0.01, 0.05],
                      #                         'n_estimators': [10,25,30,50,100,200],
                      'n_estimators': [10, 25, 30, 50, 100, 200]
                      }
        self.xgb_model = GridSearchCV(self.xgb_model, xgb_params, cv=self.k_folds,
                                      refit=True, verbose=self.grid_search_verbose, scoring='roc_auc')

        self.lbl_encoder = LabelEncoder()
        self.y_train = self.lbl_encoder.fit_transform(self.y_train)
        self.xgb_model.fit(self.X_train, self.y_train)
        print("Best params: ", self.xgb_model.best_params_)
        self.xgb_model_best_params = self.xgb_model.best_params_
        self.xgb_model = XGBClassifier(**self.xgb_model.best_params_).fit(self.X_train, self.y_train)
        self.predicted_categories = self.xgb_model.predict(self.X_test)
        self.predicted_categories = self.lbl_encoder.inverse_transform(self.predicted_categories)
        joblib.dump(self.xgb_model, Path(self.model_save_path).joinpath('xgb_model.pkl'))

    def choose_best_model(self):
        """
        Method to pick the best model out from the above set of algorithms based on accuracy
        :return: best model name
        """

        if self.model_metrics != {}:
            pass
        else:
            with open(Path(self.model_save_path).joinpath('model_performance_metrics.json'), 'r') as f:
                self.model_metrics = json.load(f)

        accuracy = {}
        for k, v in self.model_metrics.items():
            accuracy[k] = self.model_metrics[k]['model_accuracy']
        self.best_model = max(accuracy.items(), key=lambda k: k[1])[0]
        print("The best model chosen based on train performance for inference is: ", self.best_model)

    def ensemble_model(self):
        """
        Method to build a model with naive bayes, decision tree, random forest and xgb and predicting the
        outcome via majority voting
        :return: pickel model file
        """
        models = [('nb', MultinomialNB(**self.nb_model_best_params)),
                  ('dt', DecisionTreeClassifier(**self.dt_model_best_params)),
                  ('rf', RandomForestClassifier(**self.rf_model_best_params)),
                  ('xgb', XGBClassifier(**self.xgb_model_best_params))]
        self.ensemble = VotingClassifier(estimators=models, voting='hard')
        self.ensemble = self.ensemble.fit(self.X_train, self.y_train)
        self.predicted_categories = self.ensemble.predict(self.X_test)
        self.predicted_categories = self.lbl_encoder.inverse_transform(self.predicted_categories)
        # print(self.ensemble.predict_proba(self.X_test))
        joblib.dump(self.ensemble_model, Path(self.model_save_path).joinpath('ensemble_model.pkl'))

    def model_performance(self, model_name):
        """
        To measure the performance of the trained model. Accuracy is taken as the measure of evaluation.
        :param model_name: name of the model to be evaluated for performance
        :return: json file with performance metrics
        """
        print("-------------- Model metrics --------------------")
        self.model_metrics[model_name] = {}
        self.model_metrics[model_name]["model_accuracy"] = str(np.round(
            accuracy_score(self.y_test, self.predicted_categories) * 100, 2))+"%"
        self.model_metrics[model_name]["cf_report"] = classification_report(self.y_test, self.predicted_categories)
        print("|| The accuracy of the {} model is:{} ".format(model_name,
                                                              self.model_metrics[model_name]["model_accuracy"],
                                                              "    ||"))
        print("|| The classification report for the {} model built is: \n\n{} ".format(model_name,
                                                                                       self.model_metrics[model_name][
                                                                                           "cf_report"], "    ||"))
        with open(Path(self.model_save_path).joinpath("model_performance_metrics.json"), "w") as outfile:
            json.dump(self.model_metrics, outfile)

    def model_explainer(self, model):
        """
        To extract the feature importance from the model to generate reasoning for the prediction
        :param model: name of the model to be explained
        :return: string explaining the reason for prediction
        """
        exp = shap.TreeExplainer(model)
        # print(self.tf_idf_model.inverse_transform(self.tf_idf_test_vector[1,:].toarray()))
        print(self.data.todense().shape)
        self.sv = exp.shap_values(self.data.todense(), check_additivity=False)
        # column_names_for_features = list(self.tf_idf_model.get_feature_names_out()) + list(self.X_train_columns)
        # print(len(column_names_for_features))
        shap.summary_plot(self.sv, self.data.todense(), feature_names=self.X_train_column_names)


    def train_val_run(self,file_path,model_save_path,grid_search_verbose=3):
        """
        To execute training on the provided data and build list of model including Naive bayes,
        Decision Tree, Random Forest, xgboost, ensemble model
        :param grid_search_verbose: int value to decide on the type of updates to print during gridsearch CV, set to False if no updates is required
        :return: model pickle files and performace metrics
        """
        self.grid_search_verbose = grid_search_verbose
        self.model_save_path = model_save_path
        with open(Path(self.model_save_path).joinpath("model_path.json"), "w") as outfile:
            json.dump(self.model_save_path, outfile)
        print("################ Training initiated ##################")
        self.read_data(file_path)
        print(" >>>   Data from the given path is loaded into memory  >>>")
        self.prep(run='train', model_save_path=self.model_save_path)
        self.train_test_split()

        self.tfidf_vectorizer('statement')
        self.tfidf_vectorizer('statement_context')
        self.merge_features(['statement', 'statement_context'])

        print("\n**************** Naive Bayes Classifier ******************")
        self.Naive_Bayes()
        self.model_performance('Naive Bayes')

        print("\n**************** Decision Tree Classifier ******************")
        self.Decision_Tree()
        self.model_performance('Decision Tree')

        print("\n**************** Random Forest Classifier ******************")
        self.Random_Forest()
        self.model_performance('Random Forest')

        print("\n**************** XGBoost Classifier ******************")
        self.xgboost()
        self.model_performance('XGBoost')

        print("\n**************** Ensemble Classifier ******************")
        self.ensemble_model()
        self.model_performance('Ensemble')


    def inference(self, test_input, model_path):
        print("################ Test initiated ##################")
        self.data = pd.DataFrame(test_input, columns=list(test_input.keys()), index=[0])
        self.model_save_path = model_path
        self.prep(run='test',model_save_path=self.model_save_path)
        self.tfidf_vectorizer('statement')
        self.tfidf_vectorizer('statement_context')
        self.merge_features(['statement', 'statement_context'])
        self.choose_best_model()

        if self.best_model == 'Naive Bayes':
            nb_model = joblib.load(Path(self.model_save_path).joinpath("nb_model.pkl"))
            self.prediction = nb_model.predict(self.data)
        elif self.best_model == 'Decision Tree':
            dt_model = joblib.load(Path(self.model_save_path).joinpath("dt_model.pkl"))
            self.prediction = dt_model.predict(self.data)
        elif self.best_model == 'Random Forest':
            rf_model = joblib.load(Path(self.model_save_path).joinpath("rf_model.pkl"))
            self.prediction = rf_model.predict(self.data)
        elif self.best_model == 'xgb_model':
            xgb_model = joblib.load(Path(self.model_save_path).joinpath("xgb_model.pkl"))
            self.prediction = xgb_model.predict(self.data)
        elif self.best_model == 'ensemble':
            ensemble_model = joblib.load(Path(self.model_save_path).joinpath("ensemble_model.pkl"))
            self.prediction = ensemble_model.predict(self.data)



