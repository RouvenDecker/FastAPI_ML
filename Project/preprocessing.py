import re
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# EN_STOPWORDS = stopwords.words("english")

def clean_sentences(sentences: list[str]) -> list[str]:
    for idx, s in enumerate(sentences):
        s = s.lower()

        s = re.sub(r"^@[\w]+", "", s)
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"https?://[\w\d/]+", "", s)
        s = re.sub(r"[\d]+", "", s)


        # s = " ".join([word for word in s.split(" ") if word not in EN_STOPWORDS])
        tokens = word_tokenize(s)

        lemmatizer = WordNetLemmatizer()
        s = " ".join([lemmatizer.lemmatize(token) for token in tokens])
        sentences[idx] = s
    return sentences


def simple_sentiment_pipeline(s: list[str]) -> list[str]:
    countvec = joblib.load("Project/ML_models/tfidf_SVC/CountVec_1.pkl")
    tfidf_transformer = joblib.load("Project/ML_models/tfidf_SVC/tfidf_transformer_1.pkl")
    svc = joblib.load("Project/ML_models/tfidf_SVC/SVC_1.pkl")

    encoding = {1: "positive", 0: "negative"}

    x = countvec.transform(s)
    x = tfidf_transformer.transform(x)
    y_preds: np.ndarray = svc.predict(x)

    return [encoding[y_pred] for y_pred in y_preds]


if __name__ == "__main__":
    s = "I`d have responded, if I were going"
    print(clean_sentences([s]))
    s2 = ["Sooo SAD I will miss you here in San Diego!!!", "my boss is bullying me..."]
    print(clean_sentences(s2))
