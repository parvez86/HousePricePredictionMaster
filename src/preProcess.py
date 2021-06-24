import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from pandas.plotting import scatter_matrix


class PreProcess:
    def __init__(self, file='../data/house_data2.csv'):
        self.house_data = pd.read_csv(file)
        del (self.house_data['Unnamed: 0'])

    def prepareData(self):
        # print(self.house_data.info())

        # handling missing values
        self.house_data['ZN'] = self.house_data['ZN'].fillna(self.house_data['ZN'].median())
        self.house_data['INDUS'] = self.house_data['INDUS'].fillna(self.house_data['INDUS'].median())
        self.house_data['NOX'] = self.house_data['NOX'].fillna(self.house_data['NOX'].median())
        self.house_data['RM'] = self.house_data['RM'].fillna(self.house_data['RM'].median())
        self.house_data['AGE'] = self.house_data['AGE'].fillna(self.house_data['AGE'].median())
        self.house_data['DIS'] = self.house_data['DIS'].fillna(self.house_data['DIS'].median())
        self.house_data['RAD'] = self.house_data['RAD'].fillna(self.house_data['RAD'].median())
        self.house_data['TAX'] = self.house_data['TAX'].fillna(self.house_data['TAX'].median())
        self.house_data['PTRATIO'] = self.house_data['PTRATIO'].fillna(self.house_data['PTRATIO'].median())
        self.house_data['B'] = self.house_data['B'].fillna(self.house_data['B'].median())
        self.house_data['CHAS'] = self.house_data['CHAS'].fillna(self.house_data['CHAS'].median())
        self.house_data['LSTAT'] = self.house_data['LSTAT'].fillna(self.house_data['LSTAT'].median())
        self.house_data['PRICE'] = self.house_data['PRICE'].fillna(self.house_data['PRICE'].median())

        # print(self.house_data.info())

        # imputer = SimpleImputer(strategy='median')
        # imputer.fit(self.house_data)
        # X = imputer.transform(self.house_data)
        # self.house_data = pd.DataFrame(X, columns=self.house_data.columns) # filling missing values
        # self.house_data = pd.DataFrame(my_pipeline.fit_transform(self.house_data), columns=self.house_data.columns)

        # using shuffle before splitting
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(self.house_data, self.house_data['CHAS']):
            strat_train_set = self.house_data.loc[train_index]
            strat_test_set = self.house_data.loc[test_index]


        strat_train_set.to_csv('../data/trainData/train_set.csv')
        strat_test_set.to_csv('../data/trainData/test_set.csv')

    def getData(self):
        self.train_set = pd.read_csv('../data/trainData/train_set.csv')
        self.test_set = pd.read_csv('../data/trainData/test_set.csv')
        del (self.train_set['Unnamed: 0'])
        del (self.test_set['Unnamed: 0'])
        return self.train_set, self.test_set

    def get_corr(self):
        corr_matrix = self.house_data.corr()
        # print(corr_matrix['PRICE'].sort_values(ascending=False))
        # self.house_data.hist(bins=50, figsize=(20, 20))
        scatter_matrix(self.house_data, figsize=(20, 20))
        plt.savefig('../data/FeaturesCorrelation.png', bbox_inches='tight')

    def get_splitData(self):
        house_data = self.house_data
        # del (house_data['Unnamed: 0'])
        Y = house_data['PRICE']
        house_data = house_data.drop('PRICE', axis=1)
        features = house_data.columns
        # fig, ax = plt.subplots(4, 4, sharex='col')
        # fig.subplots_adjust(bottom=0.15, left=0.2)

        fig = plt.figure(figsize=(16,16))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i in range(13):
            ax = fig.add_subplot(4, 4, i+1)
            ax.scatter(house_data[features[i]], Y, color='r')
            ax.set_xlabel(features[i])
            ax.set_ylabel('PRICE')
        plt.savefig('../data/HouseData.png', bbox_inches='tight')
        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        y = self.train_set['PRICE']
        for i in range(13):
            ax = fig.add_subplot(4, 4, i + 1)
            ax.scatter(self.train_set[features[i]], y, color='r')
            ax.set_xlabel(features[i])
            ax.set_ylabel('PRICE')
        plt.savefig('../data/TrainData.png', bbox_inches='tight')

        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        y = self.test_set['PRICE']
        for i in range(13):
            ax = fig.add_subplot(4, 4, i + 1)
            ax.scatter(self.test_set[features[i]], y, color='r')
            ax.set_xlabel(features[i])
            ax.set_ylabel('PRICE')
        plt.savefig('../data/TestData.png', bbox_inches='tight')
        plt.close()


x = PreProcess()
x.prepareData()
x.getData()
x.get_splitData()
x.get_corr()
