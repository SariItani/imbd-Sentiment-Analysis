import pandas as pd
from bs4 import BeautifulSoup
import re
import joblib
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

classifier = joblib.load('../models/logistic_regression_model.pkl')
cv = joblib.load('../models/count_vectorizer.pkl')

unlabeled_data = pd.read_csv('../data/unlabeledTrainData.tsv',delimiter='\t')
unlabeled_data.head()

corpus = []
for i in range(0, len(unlabeled_data)):
    letter = BeautifulSoup(unlabeled_data['review'][i]).get_text()
    r = re.sub('[^a-zA-Z]',' ',letter)
    r = r.lower()
    r = r.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    r = [ps.stem(word) for word in r if not word in set(all_stopwords)]
    r = ' '.join(r)
    corpus.append(r)

x_unlabeled = cv.transform(corpus).toarray()
predictions = classifier.predict(x_unlabeled)

prediction_df = pd.DataFrame({'id': unlabeled_data['id'], 'sentiment': predictions})
prediction_df.to_csv('../data/unlabeled_predictions.csv', index=False)
print("Predictions saved successfully!")
