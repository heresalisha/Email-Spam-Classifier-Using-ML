import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

app = Flask(__name__)


def load_and_train_model():
    encoder = LabelEncoder()

    # Load the dataset from Excel
    df = pd.read_excel("mail_data.xlsx")

    # Replace NaN values with empty strings
    df = df.fillna("")

    df["Category"] = df["Category"].apply(lambda x: x.strip())

    # Map 'spam' to 1 and 'ham' to 0 in the 'Category' column
    df["Category"] = encoder.fit_transform(df["Category"])

    # Split the data into features (X) and target labels (Y)
    X = df["Message"].astype(str)
    Y = df["Category"]

    # Split the data into training and testing sets
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Create a TF-IDF vectorizer
    feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

    # Fit and transform the training data
    X_train_features = feature_extraction.fit_transform(X_train)

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)

    return model, feature_extraction


model, feature_extraction = load_and_train_model()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_message = request.form["message"]

    message_tfidf = feature_extraction.transform([user_message])
    prediction = model.predict(message_tfidf)[0]
    confidence_score = model.predict_proba(message_tfidf).max()

    pred = ""
    isspam = ""
    if prediction == 1:
        pred = "spam"
        isspam = True
        print("spam")
    else:
        pred = "ham"
        isspam = False
        print("ham")
    print(confidence_score)

    return render_template(
        "result.html",
        user_message=user_message,
        prediction=pred,
        isspam=isspam,
        confidence=round(confidence_score * 100, 2),
    )


if __name__ == "__main__":
    app.run(debug=True)
