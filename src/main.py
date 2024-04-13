from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import auc, confusion_matrix,accuracy_score, classification_report, roc_curve
import joblib

# load data
train = pd.read_csv('../data/labeledTrainData.tsv', delimiter='\t')
train.head()

test = pd.read_csv('../data/testData.tsv',delimiter="\t")
test.head()

# stemming and stuff for train
corpus_train = []
for i in range(0, len(train)):
    letter = BeautifulSoup(train['review'][i]).get_text()
    r = re.sub('[^a-zA-Z]',' ',letter)
    r = r.lower()
    r = r.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    r = [ps.stem(word) for word in r if not word in set(all_stopwords)]
    r = ' '.join(r)
    corpus_train.append(r)
    
# stemming and stuff for test
corpus_test = []
for i in range(0, len(test)):
    letter = BeautifulSoup(test['review'][i]).get_text()
    r = re.sub('[^a-zA-Z]',' ',letter)
    r = r.lower()
    r = r.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    r = [ps.stem(word) for word in r if not word in set(all_stopwords)]
    r = ' '.join(r)
    corpus_test.append(r)
    
# Training Yay
cv = CountVectorizer(max_features=1250)
x = cv.fit_transform(corpus_train).toarray()
y = train['sentiment'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

np.concatenate((y_test.reshape((len(y_test),1)),y_pred.reshape((len(y_pred),1))),1)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# Evaluation
new_x_test = cv.transform(corpus_test).toarray()
new_y_pred = classifier.predict(new_x_test)
print(new_y_pred)

sub = pd.DataFrame(data={'id':test['id'], 'sentiment':new_y_pred})

sub.to_csv('../data/sub.csv',index=False)
print(sub)


# Word Cloud
def plot_word_cloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('../results/WordCloud.png')
plot_word_cloud(train['review'])

# Histogram of review lengths
train['review_length'] = train['review'].apply(len)
plt.figure(figsize=(10, 5))
plt.hist(train['review_length'], bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.savefig('../results/Review_Lengths.png')

# Bar plot of class distribution
plt.figure(figsize=(6, 4))
train['sentiment'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('../results/Class_distribution.png')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('../results/Confusion_matrix.png')

# Receiver Operating Characteristic (ROC) Curve
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(x_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('../results/ROC_Curve.png')

# Saving Classification Report
with open('../results/classification_report.txt', 'w') as f:
  f.write(str(classification_report(y_test, y_pred)))
print("Classification Report saved to ../results/classification_report.txt")

# Save model
joblib.dump(classifier, '../models/logistic_regression_model.pkl')

# Save CountVectorizer
joblib.dump(cv, '../models/count_vectorizer.pkl')

# Save evaluation metrics to a text file
with open('../results/evaluation_metrics.txt', 'w') as file:
    file.write("Accuracy: {}\n".format(accuracy_score(y_test, y_pred)))
    file.write("Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, y_pred)))
    file.write("Classification Report:\n{}".format(classification_report(y_test, y_pred)))
