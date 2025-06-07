import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from preprocessing import (
    preprocessing,
    convert_text_to_vector,
    convert_text_to_label
)

data_dir = os.environ.get('DATA_DIR', os.getcwd())

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    inputs = df['text'].values.tolist()
    labels = df['sentiment'].values.tolist()
    return inputs, labels

if __name__ == '__main__':
    twitter_file = os.path.join(data_dir, 'cleaned_twitter_data.csv')
    guardian_file = os.path.join(data_dir, 'cleaned_guardian_df.csv')

    try:
        # Load and process data
        inputs_twitter, labels_twitter = load_and_process_data(twitter_file)
        inputs_guardian, labels_guardian = load_and_process_data(guardian_file)

        inputs = inputs_twitter + inputs_guardian
        labels = labels_twitter + labels_guardian

        inputs = list(map(preprocessing, inputs))

        data = convert_text_to_vector(inputs)
        labels = convert_text_to_label(labels)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Train a Random Forest Classifier
        model = RandomForestClassifier(n_estimators=10000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Random Forest Accuracy: {accuracy}")
        print("\nRandom Forest Classification Report:\n", report)

        # Try a Support Vector Machine
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train, y_train)

        y_pred_svm = svm_model.predict(X_test)

        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        report_svm = classification_report(y_test, y_pred_svm)

        print(f"SVM Accuracy: {accuracy_svm}")
        print("\nSVM Classification Report:\n", report_svm)

        f1 = f1_score(y_test, y_pred_svm, average='weighted')
        precision = precision_score(y_test, y_pred_svm, average='weighted')
        recall = recall_score(y_test, y_pred_svm, average='weighted')
        print(f'SVM F1 Score: {f1:.4f}')
        print(f'SVM Precision: {precision:.4f}')
        print(f'SVM Recall: {recall:.4f}')

        # Try a Naive Bayes Classifier
        param_grid = {
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3),
        }

        pipeline = Pipeline([
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_accuracy = grid_search.best_score_

        print(f"Best Naive Bayes Model: {best_model}")
        print(f"Best Naive Bayes Accuracy: {best_accuracy}")

        y_pred_best = best_model.predict(X_test)
        report_best = classification_report(y_test, y_pred_best)
        print("\nBest Naive Bayes Model Classification Report:\n", report_best)

        f1 = f1_score(y_test, y_pred_best, average='weighted')
        precision = precision_score(y_test, y_pred_best, average='weighted')
        recall = recall_score(y_test, y_pred_best, average='weighted')
        print(f'Best Naive Bayes F1 Score: {f1:.4f}')
        print(f'Best Naive Bayes Precision: {precision:.4f}')
        print(f'Best Naive Bayes Recall: {recall:.4f}')

    except FileNotFoundError as e:
        print(f"Error: One of the input files was not found. {e}")
