#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import graphviz
from sklearn.model_selection import train_test_split
import scipy.optimize
import operator
from sklearn import tree

def gini(df, category_name, category_values, label_name, labels_values):
    category_cardinalities = { 
        category : len(df.loc[df[category_name] == category]) for category in category_values }
    labels_cardinalities = dict()
    for category in category_values:
        for label in labels_values:
            labels_cardinalities[(category, label)] = \
                len(df.loc[ (df[category_name] == category) & (df[label_name] == label) ] )
    gini = 0
    for category in category_values:
        s = 0
        for label in labels_values:
            s += labels_cardinalities[(category, label)]**2
        gini += category_cardinalities[category]/float(len(df))*(1 - s/float(category_cardinalities[category])**2)
    return gini    

def gini_by_age(df, t):
    df['age_group'] = df['age'].apply(lambda row : 'child' if row <= t else 'adult')
    g = gini(df, 'age_group', ['child', 'adult'], 'survived', [0,1])
    return g

def plot_gini_age(df, t0, t1, step, fname):
    ages = np.arange(t0, t1, step)
    y = [gini_by_age(titanic_df, a) for a in ages]
    plt.xlabel('age')
    plt.ylabel('gini')
    plt.scatter(ages, y, c='orange')
    plt.savefig(fname)
    min_index, min_value = min(enumerate(y), key=operator.itemgetter(1))
    return ages[min_index], min_value

def use_tree(df, fname):
    X = np.array(df['age']).reshape((len(df['age']),1))
    y = df['survived']
    clf = tree.DecisionTreeClassifier(max_depth=1).fit(X,y)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(fname)


titanic_df = pd.read_csv("titanic_ds.csv")
#check the gini index of the covariant sex
sex_cov = titanic_df[['sex', 'survived']]
print gini(sex_cov, 'sex', ['female', 'male'], 'survived',[0,1])
# Now check the age
age_cov = titanic_df[['age', 'survived']].dropna()
age, min_index = plot_gini_age(age_cov, 4, 20, 0.2, '../../book/gini_age.eps')
print "At age %f, the Gini Index is %f" %(age, min_index)
use_tree(age_cov, '../../book/graph.pdf')







