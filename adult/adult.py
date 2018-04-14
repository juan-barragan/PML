import csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc

def tokenize(file_name):
    all_words = []
    with open(file_name) as f:
        tokens = dict()
        counter = 0
        reader = csv.reader(f)
        for row in reader:
            current_row_values = []
            for entry in row:
                if entry == ' <=50K':
                    current_row_values.append(0)
                    continue
                if entry == ' >50K':
                    current_row_values.append(1)
                    continue
                word = ''.join(l for l in entry if l.isalnum()).upper()
                try:
                    current_row_values.append(tokens[word])
                except: 
                    tokens[word] = counter
                    current_row_values.append(counter)
                    counter += 1
            all_words.append(current_row_values)
                
    return tokens, pd.DataFrame(all_words)

def tokenize_test_data(tokens, file_name):
    all_words = []
    with open(file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            current_row_values = []
            for entry in row:
                if entry == ' <=50K':
                    current_row_values.append(0)
                    continue
                if entry == ' >50K':
                    current_row_values.append(1)
                    continue
                word = ''.join(l for l in entry if l.isalnum()).upper()
                current_row_values.append(tokens[word])
            all_words.append(current_row_values)
                
    return tokens, pd.DataFrame(all_words)

tokens, df = tokenize('adult.data')
y = df[df.columns[df.shape[1]-1]]
df.drop([df.shape[1]-1], axis = 1, inplace=True)

clf = GaussianNB()
clf.fit(df, y)
yp = clf.predict(df)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


y_test = y
y_score = yp
average_precision = average_precision_score(y_test, y_score)
print average_precision
precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()