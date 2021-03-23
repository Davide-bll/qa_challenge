# models
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from time import time

def partial_classification_model(obj, x, y, k=10):
    """ Given a model object, give the score of the obj algorithm, using kfold cross validation"""
    # fit the model using train/test division
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    obj.partial_fit(X_train, y_train, classes=[0, 1])

    res = {'fitted_model': obj,
           'cv_score': cross_val_score(obj, x, y, cv=k).mean()}
    return res

def partial_regression_model(obj, x, y, k=10):
    """ Given a model object, give the score of the obj algorithm, using kfold cross validation"""

    # fit the model using train/test division
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    obj.partial_fit(X_train, y_train)

    res = {'fitted_model': obj,
           'cv_score': cross_val_score(obj, x, y, cv=k).mean()}

    return res

def test_partial_regression_model(obj, x, y, debug_mode=False):
    """ Given a model object, give the score of the obj algorithm, using kfold cross validation"""
    if debug_mode:
        print("inside partial regression model")
        init_time = time()

    # fit the model using train/test division
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    obj.partial_fit(X_train, y_train)

    res = {'fitted_model': obj, 'cv_score': cross_val_score(obj, x, y, cv=10).mean()}

    if debug_mode:
        dtime = time() - init_time
        print("partial results before cv obtained in " + str(dtime))

    return res


def offline_regression_model(obj, x, y, k=10):
    """ Given a model object, give the score of the obj algorithm, using kfold cross validation"""
    # fit the model using train/test division
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    obj.fit(X_train, y_train)

    res = {'fitted_model': obj,
           'cv_score': cross_val_score(obj, x, y, cv=k).mean()}
    return res

def classification_model(obj, x, y, k=10):
    """ Given a model object, give the score of the obj algorithm, using kfold cross validation"""
    # fit the model using train/test division
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    obj.fit(X_train, y_train)

    res = {'fitted_model': obj,
           'cv_score': cross_val_score(obj, x, y, cv=k).mean()}
    return res


def classification_algo(list_models, x, y):
    """apply classification with a list of models"""
    return list(map(lambda obj: classification_model(obj, x, y), list_models))
