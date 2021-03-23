import sqlite3 as sqlite
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
# models
from sklearn.linear_model import SGDClassifier

# defined modules
from api.helpers_datavis import plot_classification_boundaries
from api.helpers_models import classification_algo, partial_classification_model


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
# slice 10000 raws for testing
df = df[1:100000]

pl_box = df.plot(kind='box', subplots=True, title='Box Plots')
pl_hist = df.plot(kind='hist', subplots=True, title='Hist Plots')

features_name = ['competence', 'network_ability']
X_all = df[features_name]
y_all = df['promoted']

# parameters
k = 10  # 10 k cross validation

# Classification model
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state=0)

features_name = ['competence', 'network_ability']

mdl = SGDClassifier()

for i in range(0, 10):
    idx = 100*i
    print("Analazing data from " + str(idx) + " to " + str(idx+100))
    # read stream
    df_stream = df[idx:(idx+100)]
    # X and Y
    X = df_stream[features_name]
    y = df_stream['promoted']
    mdl.partial_fit(X=X, y=y, classes=[0, 1])


plot_classification_boundaries(X_all, y_all, features=features_name, model_obj=mdl)
