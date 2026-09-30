import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_spam_model(csv_path="mail_data.csv"):
    raw = pd.read_csv(csv_path)
    data = raw.where(pd.notnull(raw), "")
    data["Category"] = data["Category"].map({"spam":0, "ham":1})
    X = data["Message"]
    y = data["Category"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3,stratify=y)
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_features,y_train)
    return data, vectorizer, model, X_train, X_test, y_train, y_test, X_train_features, X_test_features

def predict_mail(message, vectorizer, model):
    features = vectorizer.transform([message])
    pred = int(model.predict(features)[0])
    return "ham" if pred==1 else "spam"

if __name__ == "__main__":
    data, vectorizer, model, X_train, X_test, y_train, y_test, Xtr, Xte = train_spam_model()
    print("Test prediction:", predict_mail("Congratulations! You have won a free prize. Call now!", vectorizer, model))
