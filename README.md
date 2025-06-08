# Australian Immigration Sentiment Analysis

## Overview

This project analyzes public sentiment about Australian immigration by scraping and processing data from Twitter, Reddit, and The Guardian. The aim is to uncover trends, influential voices, and platform-specific sentiment variations regarding immigration-related policies and discussions.

## What I Did

- **Scraped Data** from Twitter and Reddit using Selenium and BeautifulSoup, and used The Guardian's API for news articles.
- **Transformed & Cleaned Data**:
  - Lowercased all text and removed punctuation, numbers, and special characters.
  - Tokenized text and removed stopwords.
  - Applied lemmatization/stemming.
  - Handled missing values and removed duplicates.
- **Labeled Sentiment** using rule-based and pre-trained models (e.g., VADER, TextBlob), by assigning positive, negative, or neutral labels based on polarity scores.
- **Feature Extraction** with TF-IDF vectorization and word embeddings.
- **Trained Models**:
  - Started with Random Forest and SVM (F1-score: 65%).
  - Improved performance with deep learning (CNN, F1-score: 76%).
- **Analyzed Results** and extracted insights on sentiment variation across platforms and influential contributors.

## Techniques Used

- Web scraping (Selenium, BeautifulSoup, API integration)
- NLP preprocessing (tokenization, stopwords, lemmatization)
- Sentiment analysis (VADER, TextBlob, manual annotation)
- Feature engineering (TF-IDF, word embeddings)
- Machine learning (Random Forest, SVM, CNN, LSTM)
- Model optimization (Grid Search, F1-score evaluation)

## Results & Insights

- **Sentiment Variation Across Platforms:**
  - **Twitter:** Predominantly neutral (72.16%) — balanced commentary
  - **Reddit:** Mix of neutral (56.36%) and negative (37.74%) — critical discussion
  - **The Guardian:** Heavily negative (61.34%) — problem-centric reporting

- **Key Topics:** Immigration and visa policies dominate discussions, reflecting ongoing public and media focus on student programs, visa processing, and recent government announcements in Australia (mid-2025).

- **Influential Voices on Twitter:** 
  - Top users include political figures: "AlexHawkeMP," "ScottMorrisonMP," "AlboMP," "SecMayorkas."
  - Discourse is shaped by official statements and policy updates, as seen in frequent terms like "announced," "update," and "apply."

- **Model Performance:**
  - Random Forest/SVM baseline: **65% F1-score**
  - CNN (deep learning): **76% F1-score**
  - Indicates substantial improvement with advanced models on complex, multi-platform data.
  - **Given the noisy and nuanced nature of social media data, these results are considered strong and indicative of robust model performance in a challenging context.**

## Future Work

- Incorporate transformer-based language models (e.g., BERT, LLMs) for further accuracy.
- Expand dataset and include more diverse sources.
- Explore aspect-based sentiment analysis for finer-grained insights.

---

*For questions or collaboration, contact [krystalcodess](https://github.com/krystalcodess).*
