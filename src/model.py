import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from joblib import dump
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# build a rectangle in axes coords
left, width = .25, .7
bottom, height = .25, .6
right = left + width
top = bottom + height


class Model:
    def __init__(self, train_data, test_data):
        self.train_set = train_data
        self.test_set = test_data
        self.model_labels = {
            'LiR': 'Linear Regression',
            'EN': 'ElasticNet CV',
            'DT': 'Decision Tree',
            'RF': 'Random Forest',
            'GB': 'Gradient Boosting',
            'SGD': 'Stochastic Gradient Descent',
            'ADB': 'AdaBoost',
            'KNN': 'K Nearest Neighbors',
            'SVR': 'Support Vector Regressor',
        }

    def trainModel(self):
        train_data = self.train_set.drop('PRICE', axis=1)
        train_labels = self.train_set['PRICE'].copy()

        self.models = dict()
        self.accuracy_scores = dict()
        self.cv_accuracy_scores = dict()
        self.rmse_scores = dict()

        test_data = self.test_set.drop('PRICE', axis=1)
        test_labels = self.test_set['PRICE'].copy()

        # Elastic Net
        model = ElasticNetCV()
        model.fit(train_data, train_labels)

        scores = cross_val_score(model, test_data, test_labels, scoring='neg_mean_squared_error', cv=10)
        scores1 = cross_val_score(model, test_data, test_labels, cv=10)
        rmse_scores = np.sqrt(-scores)

        self.models['EN'] = model
        self.accuracy_scores['EN'] = model.score(test_data, test_labels)
        self.cv_accuracy_scores['EN'] = scores1.mean()
        self.rmse_scores['EN'] = rmse_scores.mean()

        # Decision Tree
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), DecisionTreeRegressor())
        model.fit(train_data, train_labels)

        scores = cross_val_score(model, test_data, test_labels, scoring='neg_mean_squared_error', cv=10)
        scores1 = cross_val_score(model, test_data, test_labels, cv=10)
        rmse_scores = np.sqrt(-scores)

        self.models['DT'] = model
        self.accuracy_scores['DT'] = model.score(test_data, test_labels)
        self.cv_accuracy_scores['DT'] = scores1.mean()
        self.rmse_scores['DT'] = rmse_scores.mean()

        # Random Forest
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), RandomForestRegressor(max_depth=10, n_estimators=100))
        model.fit(train_data, train_labels)

        scores = cross_val_score(model, test_data, test_labels, scoring='neg_mean_squared_error', cv=10)
        scores1 = cross_val_score(model, test_data, test_labels,  cv=10)
        rmse_scores = np.sqrt(-scores)

        self.models['RF'] = model
        self.accuracy_scores['RF'] = model.score(test_data, test_labels)
        self.cv_accuracy_scores['RF'] = scores1.mean()
        self.rmse_scores['RF'] = rmse_scores.mean()

        # Gradient Boosting
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), GradientBoostingRegressor(n_estimators=100))
        model.fit(train_data, train_labels)

        scores1 = cross_val_score(model, test_data, test_labels, cv=10)
        scores = cross_val_score(model, test_data, test_labels, scoring='neg_mean_squared_error', cv=10)
        rmse_scores = np.sqrt(-scores)

        self.models['GB'] = model
        self.accuracy_scores['GB'] = model.score(test_data, test_labels)
        self.cv_accuracy_scores['GB'] = scores1.mean()
        self.rmse_scores['GB'] = rmse_scores.mean()

        # Stochastic Gradient Boosting
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
        model.fit(train_data, train_labels)

        scores1 = cross_val_score(model, test_data, test_labels, cv=10)
        scores = cross_val_score(model, test_data, test_labels, scoring='neg_mean_squared_error', cv=10)
        rmse_scores = np.sqrt(-scores)

        self.models['SGD'] = model
        self.accuracy_scores['SGD'] = model.score(test_data, test_labels)
        self.cv_accuracy_scores['SGD'] = scores1.mean()
        self.rmse_scores['SGD'] = rmse_scores.mean()

        # Ada Boost
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), AdaBoostRegressor(n_estimators=100))
        model.fit(train_data, train_labels)

        scores1 = cross_val_score(model, test_data, test_labels, cv=10)
        scores = cross_val_score(model, test_data, test_labels, scoring='neg_mean_squared_error', cv=10)
        rmse_scores = np.sqrt(-scores)

        self.models['ADB'] = model
        self.accuracy_scores['ADB'] = model.score(test_data, test_labels)
        self.cv_accuracy_scores['ADB'] = scores1.mean()
        self.rmse_scores['ADB'] = rmse_scores.mean()

        # KNN
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), KNeighborsRegressor(n_neighbors=5))
        model.fit(train_data, train_labels)

        scores1 = cross_val_score(model, test_data, test_labels, cv=10)
        scores = cross_val_score(model, test_data, test_labels, scoring='neg_mean_squared_error', cv=10)
        rmse_scores = np.sqrt(-scores)

        self.models['KNN'] = model
        self.accuracy_scores['KNN'] = model.score(test_data, test_labels)
        self.cv_accuracy_scores['KNN'] = scores1.mean()
        self.rmse_scores['KNN'] = rmse_scores.mean()

        # Support Vector Regressor
        model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), SVR(kernel='sigmoid',max_iter=1000, tol=1e-3))
        model.fit(train_data, train_labels)

        scores1 = cross_val_score(model, test_data, test_labels, cv=10)
        scores = cross_val_score(model, test_data, test_labels, scoring='neg_mean_squared_error', cv=10)
        rmse_scores = np.sqrt(-scores)

        self.models['SVR'] = model
        self.accuracy_scores['SVR'] = model.score(test_data, test_labels)
        self.cv_accuracy_scores['SVR'] = scores1.mean()
        self.rmse_scores['SVR'] = rmse_scores.mean()

    def evaluate_model(self):
        # print(self.accuracy_scores)
        # print(self.cv_accuracy_scores)
        # print(self.rmse_scores)
        self.best_model = min(self.rmse_scores, key=self.rmse_scores.get)
        # print(self.best_model)

    def save_model(self):
        # Accuracy score
        keys = [k for k in self.accuracy_scores.keys()]
        values = [k for k in self.accuracy_scores.values()]
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        students = [23, 17, 35, 29, 12]

        # ax.add_patch(p)
        ax.bar(keys, values)
        for x, y in zip(keys, values):
            label = "{:.3f}".format(y)
            plt.annotate(label,  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')
        plt.title('Model Accuracy Scores')
        plt.xlabel('ML models')
        plt.ylabel('Accuracy score')
        plt.ylim(0, 1.2)
        # plt.text(1.2, 0.8, "Best Model: "+self.model_labels[self.best_model])
        ax.text(right, top, "Best Model: "+self.best_model,
                horizontalalignment='right',
                verticalalignment='bottom',
                color='r',
                transform=ax.transAxes)
        fig.savefig('../data/ModelAccuracy.png', bbox_inches='tight')
        plt.close()

        # Model Accuracy with Cross-Validation
        keys = [k for k in self.cv_accuracy_scores.keys()]
        values = [k for k in self.cv_accuracy_scores.values()]
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        students = [23, 17, 35, 29, 12]
        ax.bar(keys, values, color='orange')
        for x, y in zip(keys, values):
            label = "{:.3f}".format(y)
            plt.annotate(label,  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')
        plt.title('Model Accuracy After 10 Cross-Validation')
        plt.xlabel('ML models')
        ax.text(right, top, "Best Model: " + self.best_model,
                horizontalalignment='right',
                verticalalignment='bottom',
                color='k',
                transform=ax.transAxes)
        plt.ylabel('accuracy score')
        plt.ylim(0, 1.2)
        fig.savefig('../data/ModelCVAccuracy.png', bbox_inches='tight')
        plt.close()

        # Model RMSE
        keys = [k for k in self.rmse_scores.keys()]
        values = [k for k in self.rmse_scores.values()]
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        students = [23, 17, 35, 29, 12]
        ax.bar(keys, values, color='r')
        for x, y in zip(keys, values):
            label = "{:.3f}".format(y)
            plt.annotate(label,  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')
        plt.title('Model Root Mean Square Errors')
        plt.xlabel('ML models')
        plt.ylim(0, 10)
        ax.text(right, top, "Best Model: " + self.best_model,
                horizontalalignment='right',
                verticalalignment='bottom',
                color='k',
                transform=ax.transAxes)
        plt.ylabel('RMSE score')
        fig.savefig('../data/ModelRMSE.png', bbox_inches='tight')
        plt.close()

        # Model Comparison
        X = [k for k in self.accuracy_scores.keys()]
        accuracy_vals = [k for k in self.accuracy_scores.values()]
        cv_accuracy_vals = [k for k in self.cv_accuracy_scores.values()]
        rmse_vals = [k for k in self.rmse_scores.values()]

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        X_axis = np.arange(len(X))

        ax.bar(X_axis - 0.2, accuracy_vals, 0.4, label='Accuracy')
        ax.bar(X_axis + 0.2, cv_accuracy_vals, 0.4, label='Accuacy after Cross-validation(10)')
        for x, y in zip(X_axis, accuracy_vals):
            label = "{:.3f}".format(y)
            plt.annotate(label,  # this is the text
                         (x-0.2, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')
        for x, y in zip(X_axis, cv_accuracy_vals):
            label = "{:.3f}".format(y)
            plt.annotate(label,  # this is the text
                         (x+0.2, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')
        x_labels = [self.model_labels[k] for k in X]
        plt.xticks(X_axis, x_labels, rotation=65)
        # plt.yticks(np.arange(1, 5, step=0.5))
        plt.xlabel("ML Models")
        plt.ylabel("Score")
        plt.ylim(0, 1.2)
        ax.text(right, top, "Best Model: " + self.model_labels[self.best_model],
                horizontalalignment='right',
                verticalalignment='bottom',
                color='r',
                transform=ax.transAxes)
        plt.title("Machine Learning Model Comparison")
        plt.legend()
        fig.savefig('../data/ModelComparison.png', bbox_inches='tight')
        plt.close()
        dump(self.models[self.best_model], 'HousePricePredictor.joblib')


if __name__ == '__main__':
    train_set = pd.read_csv('../data/trainData/train_set.csv')
    test_set = pd.read_csv('../data/trainData/test_set.csv')
    del (train_set['Unnamed: 0'])
    del (test_set['Unnamed: 0'])
    X = Model(train_set, test_set)
    X.trainModel()
    X.evaluate_model()
    X.save_model()
