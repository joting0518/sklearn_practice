from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv('spam.csv')
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

messages = data['Message']

tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='unicode',
                                   token_pattern=r'\w{1,}', max_features=3000)

X_tfidf = tfidf_vectorizer.fit_transform(messages)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Category'])

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

