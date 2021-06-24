from joblib import load
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.model import Model
from src.preProcess import PreProcess
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    X = PreProcess()
    X.prepareData()
    train_set, test_set = X.getData()

    Y = Model(train_data=train_set, test_data=test_set)
    Y.trainModel()
    Y.evaluate_model()
    Y.save_model()

    data = pd.DataFrame([
        [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98],
        [0.02731, 0.0, 7.07, 0, 0.469, 6.421, 78.9, 4.9671, 2, 242, 17.8, 396.9, 9.14],
        [0.80271, 0.0, 8.14, 0, 0.538, 5.456, 36.6, 3.7965, 4, 307, 21.0, 288.99, 11.69],
        [0.08829, 12.5, 7.87, 0, 0.524, 6.012, 66.6, 5.5605, 5, 305, 15.2, 395.6, 12.43],
        [0.14455, 12.5, 7.87, 0, 0.524, 6.172, 96.1, 5.9505, 5, 300, 15.2, 396.9, 19.15],
        [0.21124, 12.5, 7.87, 0, 0.524, 5.631, 100.0, 6.0821, 5, 311, 15.2, 386.63, 29.93]
    ])
    data_label = [24.0, 21.6, 20.2, 22.9, 27.1, 16.5]
    my_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])
    data = pd.DataFrame(my_pipeline.fit_transform(data), columns=data.columns)
    model = load('HousePricePredictor.joblib')
    predict = model.predict(data).tolist()
    print("Best Model: ", Y.model_labels[Y.best_model])
    print('Prediction: ', predict)
    print('Actual value: ', data_label)
    print("RMSE: ", mean_squared_error(data_label, predict, squared=False))
