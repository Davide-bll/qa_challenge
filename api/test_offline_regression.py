import sqlite3 as sqlite
import pandas as pd
# models
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
import requests
# defined modules
from api.helpers_models import offline_regression_model, test_partial_regression_model, partial_regression_model

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# get all data
query = "SELECT * FROM data"
conn = sqlite.connect('data/main.db')
conn.row_factory = dict_factory
cur = conn.cursor()
results = cur.execute(query).fetchall()

# TO DATAFRAME
db = pd.DataFrame(results)
df = db[['competence', 'network_ability', 'promoted']]
y = df['network_ability']
X = df[['competence', 'promoted']]

loss_f = ['squared_loss', 'huber', 'epsilon_insensitive']
mdl_sl = SGDRegressor()
mdl_h = SGDRegressor(loss=loss_f[1])
mdl_ei = SGDRegressor(loss=loss_f[2])

#parameters
offline_test = False
online_sim = not offline_test

if offline_test:
    model_list = [mdl_sl, mdl_h, mdl_ei, PassiveAggressiveRegressor()]
    list_res = list(map(lambda obj: offline_regression_model(obj, X, y), model_list))
    print("list of results")
    print(list_res)
    # R^2 about 0.18 using all the data

# simulate online regression for the first 5 pages
# using the df
if online_sim:
    pages = range(0, 5)
    partial_model_list2 = [SGDRegressor(), SGDRegressor(loss=loss_f[1]), SGDRegressor(loss=loss_f[2])]
    partial_scores2 = []
    debug_mode = True
    for page in pages:
        # get Stream of data
        stream = df[100*page:100*(page + 1)]
        X, y = stream[['competence', 'promoted']], stream['network_ability']
        partial_res2 = list(map(lambda obj: partial_regression_model(obj=obj, x=X, y=y), partial_model_list2))
        partial_model_list2 = [val['fitted_model'] for val in partial_res2]
        partial_scores2.append([val['cv_score'] for val in partial_res2])

# using the api
if online_sim:
    pages = range(0, 5)
    partial_model_list = [mdl_sl, mdl_h, mdl_ei]
    partial_scores = []
    debug_mode = True
    for page in pages:
        # get Stream of data
        api_request = 'http://127.0.0.1:5000/api/v1/data?page=' + str(page)
        stream = pd.DataFrame(requests.get(api_request).json())
        X, y = stream[['competence', 'promoted']], stream['network_ability']
        partial_res = list(map(lambda obj: partial_regression_model(obj=obj, x=X, y=y), partial_model_list))
        partial_model_list = [val['fitted_model'] for val in partial_res]
        partial_scores.append([val['cv_score'] for val in partial_res])


# partials scores using df
print(partial_scores)

# partials scores using df
print(partial_scores2)
