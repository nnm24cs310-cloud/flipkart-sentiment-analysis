# Requirements Document

## Introduction

This document defines the requirements for a Customer Sentiment Analysis system applied to Flipkart product reviews. The system ingests a dataset containing product names, customer review text, and star ratings (1–5), then performs exploratory data analysis, text preprocessing, sentiment labeling, visualization, and optionally trains a machine learning classifier to predict sentiment from review text. The goal is to understand how customer sentiment correlates with product ratings and to surface actionable insights from unstructured review data.

## Glossary

- **System**: The Python-based sentiment analysis pipeline implemented in `project.py`
- **Dataset**: A CSV file containing Flipkart product reviews with columns `Product_name`, `Review`, and `Rating`
- **Product_name**: The name of the Flipkart product being reviewed
- **Review**: Free-form text written by a customer describing their experience with a product
- **Rating**: An integer from 1 to 5 representing the customer's star rating for a product
- **Sentiment_Label**: A categorical label assigned to each review — one of `Positive`, `Negative`, or `Neutral`
- **EDA**: Exploratory Data Analysis — statistical and visual examination of the dataset before modeling
- **Preprocessor**: The component responsible for cleaning and normalizing raw review text
- **Sentiment_Labeler**: The component that assigns a `Sentiment_Label` to each review based on its `Rating`
- **Visualizer**: The component that generates charts and plots from the processed data
- **Classifier**: An optional machine learning model trained to predict `Sentiment_Label` from review text
- **Stopwords**: Common words (e.g., "the", "is") that carry little semantic meaning and are removed during preprocessing
- **TF-IDF**: Term Frequency–Inverse Document Frequency — a numerical representation of text used as input to the Classifier
- **Corpus**: The full collection of preprocessed review texts used for vectorization and modeling

---

## Requirements

### Requirement 1: Data Loading

**User Story:** As a data analyst, I want to load the Flipkart reviews dataset from a CSV file, so that I can begin analysis on structured review data.

#### Acceptance Criteria

1. THE System SHALL load the dataset from a CSV file path specified at the top of `project.py` as a configurable constant.
2. WHEN the CSV file is loaded successfully, THE System SHALL display the shape of the DataFrame (number of rows and columns).
3. WHEN the CSV file is loaded successfully, THE System SHALL display the first five rows of the DataFrame.
4. IF the CSV file path does not exist or cannot be read, THEN THE System SHALL print a descriptive error message and halt execution.
5. THE System SHALL verify that the loaded DataFrame contains the columns `Product_name`, `Review`, and `Rating`; IF any column is missing, THEN THE System SHALL print a descriptive error message and halt execution.

---

### Requirement 2: Exploratory Data Analysis (EDA)

**User Story:** As a data analyst, I want to explore the dataset's structure and distributions, so that I can understand data quality and key patterns before modeling.

#### Acceptance Criteria

1. THE System SHALL display summary statistics (count, mean, std, min, max, quartiles) for the `Rating` column.
2. THE System SHALL display the count of missing values for each column in the DataFrame.
3. WHEN missing values are present in the `Review` column, THE System SHALL drop those rows and log the number of rows removed.
4. WHEN missing values are present in the `Rating` column, THE System SHALL drop those rows and log the number of rows removed.
5. THE System SHALL display the distribution of `Rating` values as a count of each unique rating (1 through 5).
6. THE System SHALL display the top 10 most frequently reviewed products by review count.

---

### Requirement 3: Text Preprocessing

**User Story:** As a data scientist, I want to clean and normalize review text, so that downstream sentiment analysis and modeling operate on consistent, noise-free input.

#### Acceptance Criteria

1. THE Preprocessor SHALL convert all characters in the `Review` column to lowercase.
2. THE Preprocessor SHALL remove all URLs from review text.
3. THE Preprocessor SHALL remove all HTML tags from review text.
4. THE Preprocessor SHALL remove all punctuation and special characters from review text, retaining only alphabetic characters and whitespace.
5. THE Preprocessor SHALL remove Stopwords from review text using the NLTK English stopwords list.
6. THE Preprocessor SHALL apply stemming or lemmatization to reduce words to their base form.
7. WHEN preprocessing is complete, THE System SHALL store the cleaned text in a new column named `Cleaned_Review` in the DataFrame.
8. THE System SHALL display the first five rows of the `Review` and `Cleaned_Review` columns side-by-side for verification.

---

### Requirement 4: Sentiment Labeling

**User Story:** As a data analyst, I want each review to be assigned a sentiment label based on its star rating, so that I can use ratings as a ground-truth proxy for sentiment.

#### Acceptance Criteria

1. THE Sentiment_Labeler SHALL assign the label `Positive` to reviews where `Rating` is 4 or 5.
2. THE Sentiment_Labeler SHALL assign the label `Negative` to reviews where `Rating` is 1 or 2.
3. THE Sentiment_Labeler SHALL assign the label `Neutral` to reviews where `Rating` is 3.
4. THE System SHALL store the assigned label in a new column named `Sentiment_Label` in the DataFrame.
5. THE System SHALL display the count of reviews for each `Sentiment_Label` value after labeling.

---

### Requirement 5: Visualization

**User Story:** As a data analyst, I want visual representations of the data and sentiment distributions, so that I can communicate findings clearly to stakeholders.

#### Acceptance Criteria

1. THE Visualizer SHALL generate a bar chart showing the count of reviews for each `Rating` value (1 through 5).
2. THE Visualizer SHALL generate a pie chart showing the proportion of `Positive`, `Negative`, and `Neutral` reviews.
3. THE Visualizer SHALL generate a bar chart showing the top 10 most reviewed products by review count.
4. THE Visualizer SHALL generate a word cloud image from the `Cleaned_Review` text of `Positive` reviews.
5. THE Visualizer SHALL generate a word cloud image from the `Cleaned_Review` text of `Negative` reviews.
6. WHEN generating any chart, THE Visualizer SHALL include a descriptive title, axis labels where applicable, and a legend where applicable.
7. THE Visualizer SHALL save each generated chart as a PNG file in a subdirectory named `output/` relative to `project.py`.
8. IF the `output/` directory does not exist, THEN THE Visualizer SHALL create it before saving any files.

---

### Requirement 6: Sentiment–Rating Correlation Analysis

**User Story:** As a data analyst, I want to examine the relationship between numeric ratings and sentiment labels, so that I can validate the labeling strategy and surface any anomalies.

#### Acceptance Criteria

1. THE System SHALL compute and display the average `Rating` for each `Sentiment_Label` group.
2. THE System SHALL generate a box plot showing the distribution of `Rating` values grouped by `Sentiment_Label`.
3. THE Visualizer SHALL save the box plot as a PNG file in the `output/` directory.

---

### Requirement 7: Machine Learning Classifier (Optional)

**User Story:** As a data scientist, I want to train a text classification model on the labeled reviews, so that I can predict sentiment from review text alone without relying on the rating.

#### Acceptance Criteria

1. WHERE the ML classifier is enabled, THE System SHALL vectorize the `Cleaned_Review` column using TF-IDF with a maximum of 5000 features.
2. WHERE the ML classifier is enabled, THE System SHALL split the dataset into training (80%) and test (20%) sets using a fixed random seed for reproducibility.
3. WHERE the ML classifier is enabled, THE System SHALL train a Logistic Regression model on the training set.
4. WHERE the ML classifier is enabled, THE System SHALL evaluate the trained model on the test set and display accuracy, precision, recall, and F1-score per class.
5. WHERE the ML classifier is enabled, THE Visualizer SHALL generate and save a confusion matrix heatmap as a PNG file in the `output/` directory.
6. WHERE the ML classifier is enabled, THE System SHALL display the top 10 most influential TF-IDF features for each sentiment class.
7. IF the dataset contains fewer than 50 labeled samples, THEN THE System SHALL print a warning and skip classifier training.

---

### Requirement 8: Round-Trip Text Preprocessing Consistency

**User Story:** As a data scientist, I want the preprocessing pipeline to be deterministic and idempotent, so that applying it once or multiple times produces the same result.

#### Acceptance Criteria

1. FOR ALL review texts, applying THE Preprocessor twice SHALL produce the same output as applying it once (idempotence property).
2. THE System SHALL expose the preprocessing logic as a standalone callable function `preprocess_text(text: str) -> str` so that it can be tested independently.
