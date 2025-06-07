import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocessing(text: str) -> str:
    # Remove all the special characters
    text = re.sub(r'\W', ' ', text)

    # Remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)

    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)

    # Converting to Lowercase
    text = text.lower()

    return text.lower()


def convert_text_to_vector(
        data: list[str],
        max_features: int = 2500,
        min_df: int = 7,
        max_df: float = 0.8
):
    model = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=stopwords.words('english'))

    model.fit_transform(data)

    data = model.transform(data)

    return data


def convert_text_to_label(labels: list[str]):
    model = LabelEncoder()
    model.fit(labels)

    return model.transform(labels)