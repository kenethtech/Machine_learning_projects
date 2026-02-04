from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("spam.csv")

data.rename(columns={'Category': 'target', 'Message': 'message'}, inplace=True)
head = data.head()
print(head)

data['target'] = data['target'].map({'ham':0, 'spam':1})
head = data.head()
print(head)

x, y = data['message'], data['target']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=50)

model = LogisticRegression()
model.fit(x_train, y_train)

print(f"Model Accuracy:{model.score(x_test, y_test):.2%}")

def check_spam(message):
    msg_vector = vectorizer.transform([message])
    prediction = model.predict(msg_vector)[0]

    status = "!SPAM!" if prediction == 1 else "+INBOX+"
    print(f"{message} -> {status}")

check_spam("Are you still in the meeting")
check_spam("Free entry to win $1000")
