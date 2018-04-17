#!/usr/bin/python

import pandas as pd
import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import graphviz 

# Maps each category in the dataframe and it value fields to numerical values. We use bag of words method
def get_mapper(df, categories):
	mapper = dict()
	for category in categories:
		whole_set = df[category]
		values, indices = np.unique(whole_set, return_inverse= True)
		for i, v in enumerate(values):
			v = v.upper().replace(' ', '')
			mapper[(category,v)] = i
	return mapper

# Transform all the categorical data into numerical
def transform_df(df, dico, categories):
	for category in categories:
		column = [s.upper().replace(' ','') for s in df[category]]
		column = [s.upper().replace('.','') for s in column]
		column = [dico[(category,v)] for v in column]
		df[category] = column
	return df

# Error in classification we suppose len(y) == len(y_pred)
def error(y, y_pred):
	s = 0
	for i,ip in zip(y,y_pred):
		if i!=ip: s+=1
	return float(s)/len(y)

def classifier_performance(clf, df_train, y_train, df_test, y_test):
	clf.fit(df_train, y_train)
	y_pred_train = clf.predict(df_train)
	inner_error = error(y_train, y_pred_train)
	y_pred_test = clf.predict(df_test)
	outer_error = error(y_test, y_pred_test)
	return inner_error, outer_error

def clean_transform_ds(df, categories, dico):
	del df['fnlwgt'] # We ignore this it has high variance, is numerical already
	df = transform_df(df, dico, categories)
	y = df['earnings']
	del df['earnings']
	return df, y

# load and training and test datasets
categories = ['workclass','education','marital','occupation','relationship','race','sex','country','earnings']
df_train_whole = pd.read_csv('adult.data')
dico = get_mapper(df_train_whole, categories)
df_train, y_train = clean_transform_ds(df_train_whole, categories, dico)
df_test_whole = pd.read_csv('adult.test')
df_test, y_test = clean_transform_ds(df_test_whole, categories, dico)

# Calculate performance for decision tree
clf = tree.DecisionTreeClassifier(max_depth=len(categories))
print "Deision Tree"
print "Inner Error: %f, Outer error: %f" % classifier_performance(clf, df_train, y_train, df_test, y_test)

# Calculate Performance for Naive bayes
print "Naive Bayes"
clf = GaussianNB()
print "Inner Error: %f, Outer error: %f" % classifier_performance(clf, df_train, y_train, df_test, y_test)

