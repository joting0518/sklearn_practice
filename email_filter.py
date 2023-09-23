from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('spam.csv')
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

messages = data['Message']

count_vectorizer = CountVectorizer()
word_counts = count_vectorizer.fit_transform(messages)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(word_counts)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Category'])

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(y_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)