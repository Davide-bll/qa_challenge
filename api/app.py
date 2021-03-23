from flask import Flask, request, jsonify, render_template, make_response
import json
import sqlite3 as sqlite
import os
import requests
from sklearn.linear_model import SGDClassifier, SGDRegressor
from time import time
import pandas as pd

# import defined module
# if you run this in console use api.helpers_models instead of helpers models
from helpers_models import partial_classification_model, partial_regression_model
from ingestion import get_page

# parameters
features_classification = ['competence', 'network_ability']
features_reg = ['competence', 'promoted']

# Global variables: Classification Algorithm
# page uploaded for classification
pageClass = 0
# support vector machine classifier
svm_mdl = SGDClassifier(loss='hinge')
# logistic regression
lr_mdl = SGDClassifier(loss='log')
# cumulative scores of classification
cum_class_score1, cum_class_score2 = 0, 0

# Global variables: Regression Algorithm
# page uploaded for regression
pageReg = 0
# list of model for regression
mdl_reg_list = [SGDRegressor(eta0=0.001), SGDRegressor(eta0=0.001, loss='huber')]
# cumulative score of regression
cum_reg_score1, cum_reg_score2 = 0, 0

# create App
app = Flask(__name__)


def update_score(old_score, new_score, t):
    """update score mean from time t to t+1"""
    return old_score * t / (t + 1) + new_score / (t + 1)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


@app.route('/api/v1/data', methods=['GET'])
def api_filter():
    query_parameters = request.args
    page = int(query_parameters.get('page'))
    query = build_query(page)
    conn = sqlite.connect('data/main.db')
    conn.row_factory = dict_factory
    cur = conn.cursor()
    results = cur.execute(query).fetchall()

    return jsonify(results)


# Route for classification
@app.route('/classification', methods=["GET", "POST"])
def main():
    return render_template('classification_template.html')


# Route for Regression
@app.route('/reg', methods=["GET", "POST"])
def mainreg():
    return render_template('regression_template.html')


def build_query(page):
    n_per_page = 100
    offset = page * n_per_page
    # Possibly vulnerable to SQL injection.
    query = "SELECT * FROM data LIMIT 100 OFFSET " + str(offset)

    return query

# route for classification data
@app.route('/data', methods=["GET", "POST"])
def data():
    # modify global variables
    global pageClass, svm_mdl, lr_mdl, cum_class_score1, cum_class_score2

    # get stream of data
    stream = get_page(pageClass)

    # increment page for the next Api call
    pageClass = pageClass + 1

    # get X and y
    X, y = stream[features_classification], stream['promoted']

    # fit models and compute score with Cross Validation
    partial_res_svm = partial_classification_model(svm_mdl, X, y)
    partial_res_lr = partial_classification_model(lr_mdl, X, y)

    # update list of models obj
    svm_mdl, lr_mdl = partial_res_svm['fitted_model'], partial_res_lr['fitted_model']

    # partial scores
    score_svm, score_lr = partial_res_svm['cv_score'], partial_res_lr['cv_score']

    # update mean cumulative score
    cum_class_score1 = update_score(cum_class_score1, score_svm, pageClass)
    cum_class_score2 = update_score(cum_class_score2, score_lr, pageClass)

    # send data
    data = [time() * 1000, score_svm, score_lr, cum_class_score1, cum_class_score2]

    # create response object
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'

    return response

# route for regression data
@app.route('/datareg', methods=["GET", "POST"])
def datareg():
    # modify global variables
    global pageReg, mdl_reg_list, cum_reg_score1, cum_reg_score2

    # get Stream of data
    stream = get_page(pageReg)

    # increment page for the next Api call
    pageReg = pageReg + 1

    X, y = stream[features_reg], stream['network_ability']

    # fit model and compute score with CV
    partial_res_list = list(map(lambda obj: partial_regression_model(obj, X, y), mdl_reg_list))

    # update models obj
    mdl_reg_list = [val['fitted_model'] for val in partial_res_list]

    # update mean cumulative scores
    cum_reg_score1 = update_score(cum_reg_score1, partial_res_list[0]['cv_score'], pageReg)
    cum_reg_score2 = update_score(cum_reg_score2, partial_res_list[1]['cv_score'], pageReg)

    # create data to send
    datareg = [time() * 1000] + [val['cv_score'] for val in partial_res_list] + [cum_reg_score1, cum_reg_score2]

    # create response object
    response = make_response(json.dumps(datareg))
    response.content_type = 'application/json'

    return response


if __name__ == '__main__':
    if os.environ.get('PORT') is not None:

        app.run(debug=True, port=os.environ.get('PORT'))
    else:
        app.run(debug=True)
