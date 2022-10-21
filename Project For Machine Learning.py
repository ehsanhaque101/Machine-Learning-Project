from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# from sklearn import datasets
# from numpy import mean
# from numpy import absolute
# from numpy import sqrt
import pandas as pd
# import statsmodels.api as sm
import sys
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import make_pipeline
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def LR(n_splits, file):
    cv = pd.read_csv(file)
    # All data preparation steps in this cell

    # converting symboling to category
    cv['symboling'] = cv['symboling'].astype('object')

    # create new column: car_company
    p = re.compile(r'\w+-?\w+')
    cv['car_company'] = cv['CarName'].apply(
        lambda x: re.findall(p, x)[0])

    # replacing misspelled car_company names
    # volkswagen
    cv.loc[(cv['car_company'] == "vw") |
           (cv['car_company'] == "vokswagen"), 'car_company'] = 'volkswagen'
    # porsche
    cv.loc[cv['car_company']
           == "porcshce", 'car_company'] = 'porsche'
    # toyota
    cv.loc[cv['car_company']
           == "toyouta", 'car_company'] = 'toyota'
    # nissan
    cv.loc[cv['car_company']
           == "Nissan", 'car_company'] = 'nissan'
    # mazda
    cv.loc[cv['car_company']
           == "maxda", 'car_company'] = 'mazda'

    # drop carname variable
    cv = cv.drop('CarName', axis=1)

    # split into X and y
    X = cv.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
                   'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
                   'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
                   'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
                   'horsepower', 'peakrpm', 'citympg', 'highwaympg',
                   'car_company']]
    y = cv['price']

    # creating dummy variables for category variables
    cv_category = X.select_dtypes(include=['object'])
    cv_category.head()

    # convert into dummies
    cv_dummy = pd.get_dummies(
        cv_category, drop_first=True)
    cv_dummy.head()

    # drop category variables
    X = X.drop(list(cv_category.columns), axis=1)

    # concat dummy variables with X
    X = pd.concat([X, cv_dummy], axis=1)

    # rescale the features
    cols = X.columns
    X = pd.DataFrame(scale(X))
    X.columns = cols

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        test_size=0.2, random_state=50)

    len(X_train.columns)

    # creating a KFold object with 5 splits
    fold = KFold(n_splits, shuffle=True, random_state=100)

    # specify range of hyperparameters
    hyper_params = [{'n_features_to_select': list(range(2, 40))}]

    # specify model
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    rfe = RFE(lm)

    # set up GridSearchCV()
    model_cv = GridSearchCV(estimator=rfe,
                            param_grid=hyper_params,
                            scoring='r2',
                            cv=fold,
                            verbose=1,
                            return_train_score=True)

    # fit the model
    model_cv.fit(X_train, y_train)

    # cv results
    pd.set_option('display.max_columns', None)
    cv_results = pd.DataFrame(model_cv.cv_results_)
    # cv_results
    file_path = 'lr.txt'
    sys.stdout = open(file_path, "w")
    print(cv_results)


if __name__ == '__main__':
    # total arguments
    n = len(sys.argv)
    print("Arguments passed: ", n)
    AlgoName = "LR"
    file = "Car.csv"
    n_splits = 6
    # parsing the command line arguments and setting the parameters
    if n > 1:
        for i in range(1, n):
            param = sys.argv[i]
            if param == "-a":
                AlgoName = sys.argv[i+1]
                if (AlgoName != "LR"):
                    print("Wrong Algorithm name. It has to be LR")
            if param == "-n":
                n_splits = int(sys.argv[i+1])

            if param == "-f":
                dataSet = sys.argv[i+1]
    # call the functions as given in the command
    if AlgoName == "LR":
        LR(n_splits, file)
