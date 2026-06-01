# Design Document: Flipkart Customer Sentiment Analysis

## Overview

This document describes the technical design for a Python-based sentiment analysis pipeline (`project.py`) that processes Flipkart product reviews. The pipeline ingests a CSV dataset, performs exploratory data analysis, cleans and normalizes review text, assigns sentiment labels from star ratings, generates visualizations, and optionally trains a Logistic Regression classifier to predict sentiment from text alone.

The entire pipeline is implemented as a single, well-structured `project.py` file in the workspace root. Each logical stage is encapsulated in a dedicated function, making the code testable, readable, and easy to extend.

### Key Design Goals

- **Determinism**: All preprocessing and modeling steps use fixed random seeds.
- **Idempotence**: The `preprocess_text` function produces the same output when applied multiple times.
- **Modularity**: Each pipeline stage (load, EDA, preprocess, label, visualize, classify) is a standalone function.
- **Fail-fast**: Missing files or required columns cause an immediate, descriptive halt.
- **Configurability**: The CSV path and ML toggle are constants at the top of `project.py`.

---

## Architecture

The pipeline follows a linear, staged data-flow architecture. Each stage transforms or enriches the central `pandas.DataFrame` and passes it to the next stage.

```mermaid
flowchart TD
    A[CSV File] -->|load_data| B[Raw DataFrame]
    B -->|run_eda| C[Cleaned DataFrame\n(no missing rows)]
    C -->|preprocess_text| D[DataFrame + Cleaned_Review]
    D -->|label_sentiment| E[DataFrame + Sentiment_Label]
    E -->|generate_visualizations| F[PNG charts in output/]
    E -->|correlation_analysis| G[Stats + box plot]
    E -->|train_classifier\n(optional)| H[Model metrics + confusion matrix]
```

### Module Structure (single file)

```
project.py
├── Constants (CSV_PATH, ENABLE_ML, RANDOM_SEED, MAX_TFIDF_FEATURES)
├── load_data(path) -> pd.DataFrame
├── run_eda(df) -> pd.DataFrame
├── preprocess_text(text: str) -> str          # standalone, testable
├── apply_preprocessing(df) -> pd.DataFrame
├── label_sentiment(df) -> pd.DataFrame
├── generate_visualizations(df) -> None
├── correlation_analysis(df) -> None
├── train_classifier(df) -> None               # guarded by ENABLE_ML
└── main()
```

---

## Components and Interfaces

### 1. Data Loader (`load_data`)

**Responsibility**: Read the CSV file and validate its structure.

```python
def load_data(path: str) -> pd.DataFrame:
    """
    Loads the dataset from `path`.
    Prints shape and first 5 rows on success.
    Prints a descriptive error and calls sys.exit(1) on failure.
    Required columns: ['Product_name', 'Review', 'Rating']
    """
```

**Behavior**:
- Uses `pd.read_csv(path)`.
- Validates presence of `Product_name`, `Review`, `Rating` columns.
- Exits with `sys.exit(1)` and a descriptive message on any failure.

---

### 2. EDA Module (`run_eda`)

**Responsibility**: Summarize the dataset and remove rows with missing critical values.

```python
def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Displays summary statistics, missing value counts, rating distribution,
    and top-10 most reviewed products.
    Drops rows with missing Review or Rating values (logs count removed).
    Returns the cleaned DataFrame.
    """
```

**Behavior**:
- Prints `df['Rating'].describe()`.
- Prints `df.isnull().sum()`.
- Drops rows where `Review` is NaN; logs count.
- Drops rows where `Rating` is NaN; logs count.
- Prints `df['Rating'].value_counts().sort_index()`.
- Prints top-10 products by `df['Product_name'].value_counts().head(10)`.

---

### 3. Text Preprocessor (`preprocess_text`, `apply_preprocessing`)

**Responsibility**: Clean and normalize raw review text.

```python
def preprocess_text(text: str) -> str:
    """
    Applies the full preprocessing pipeline to a single string:
    1. Lowercase
    2. Remove URLs (http/https/www)
    3. Remove HTML tags
    4. Remove punctuation and special characters (keep [a-z ] only)
    5. Remove NLTK English stopwords
    6. Apply WordNet lemmatization
    Returns the cleaned string.
    """

def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies preprocess_text to every row in df['Review'].
    Stores result in df['Cleaned_Review'].
    Prints first 5 rows of Review vs Cleaned_Review side-by-side.
    Returns the modified DataFrame.
    """
```

**Libraries**: `re`, `nltk` (stopwords, WordNetLemmatizer), `string`.

**NLTK Downloads** (called once at startup): `stopwords`, `wordnet`, `omw-1.4`.

---

### 4. Sentiment Labeler (`label_sentiment`)

**Responsibility**: Assign a categorical sentiment label based on the numeric rating.

```python
def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps Rating -> Sentiment_Label:
      4, 5  -> 'Positive'
      3     -> 'Neutral'
      1, 2  -> 'Negative'
    Stores result in df['Sentiment_Label'].
    Prints value counts for Sentiment_Label.
    Returns the modified DataFrame.
    """
```

**Implementation**: Uses `pd.cut` or a vectorized `np.select` / `map` for efficiency.

---

### 5. Visualizer (`generate_visualizations`, `correlation_analysis`)

**Responsibility**: Produce and save all required charts.

```python
def generate_visualizations(df: pd.DataFrame) -> None:
    """
    Generates and saves to output/:
    - rating_distribution.png   : bar chart of Rating counts
    - sentiment_distribution.png: pie chart of Sentiment_Label proportions
    - top_products.png          : bar chart of top-10 reviewed products
    - wordcloud_positive.png    : word cloud from Positive Cleaned_Review
    - wordcloud_negative.png    : word cloud from Negative Cleaned_Review
    """

def correlation_analysis(df: pd.DataFrame) -> None:
    """
    Prints average Rating per Sentiment_Label group.
    Generates and saves output/rating_by_sentiment_boxplot.png.
    """
```

**Libraries**: `matplotlib`, `seaborn`, `wordcloud`.

**Output directory**: Created with `os.makedirs('output', exist_ok=True)` before any save.

**Chart standards**: Every chart includes a title, axis labels where applicable, and a legend where applicable.

---

### 6. ML Classifier (`train_classifier`)

**Responsibility**: Optionally train and evaluate a Logistic Regression sentiment classifier.

```python
def train_classifier(df: pd.DataFrame) -> None:
    """
    Guarded by ENABLE_ML constant.
    Steps:
    1. Guard: if len(df) < 50, print warning and return.
    2. TF-IDF vectorization of Cleaned_Review (max 5000 features).
    3. Train/test split: 80/20, random_state=RANDOM_SEED.
    4. Train LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).
    5. Print classification_report (accuracy, precision, recall, F1 per class).
    6. Save confusion matrix heatmap to output/confusion_matrix.png.
    7. Print top-10 TF-IDF features per sentiment class.
    """
```

**Libraries**: `sklearn` (TfidfVectorizer, LogisticRegression, train_test_split, classification_report, confusion_matrix).

---

## Data Models

### Central DataFrame Schema

The pipeline operates on a single `pandas.DataFrame` that is progressively enriched:

| Stage | Column Added | Type | Description |
|---|---|---|---|
| Load | `Product_name` | `str` | Product name from CSV |
| Load | `Review` | `str` | Raw customer review text |
| Load | `Rating` | `int` (1–5) | Star rating |
| Preprocess | `Cleaned_Review` | `str` | Normalized, stopword-free, lemmatized text |
| Label | `Sentiment_Label` | `str` (`Positive`/`Neutral`/`Negative`) | Sentiment category |

### Sentiment Label Mapping

| Rating | Sentiment_Label |
|--------|----------------|
| 1 | Negative |
| 2 | Negative |
| 3 | Neutral |
| 4 | Positive |
| 5 | Positive |

### Configuration Constants

```python
CSV_PATH = "flipkart_reviews.csv"   # Path to input dataset
ENABLE_ML = True                     # Toggle ML classifier
RANDOM_SEED = 42                     # Fixed seed for reproducibility
MAX_TFIDF_FEATURES = 5000            # TF-IDF vocabulary size
OUTPUT_DIR = "output"                # Directory for saved charts
```

### Output Files

| File | Description |
|------|-------------|
| `output/rating_distribution.png` | Bar chart: Rating counts |
| `output/sentiment_distribution.png` | Pie chart: Sentiment proportions |
| `output/top_products.png` | Bar chart: Top-10 products |
| `output/wordcloud_positive.png` | Word cloud: Positive reviews |
| `output/wordcloud_negative.png` | Word cloud: Negative reviews |
| `output/rating_by_sentiment_boxplot.png` | Box plot: Rating by Sentiment |
| `output/confusion_matrix.png` | Heatmap: ML confusion matrix (if ENABLE_ML) |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Preprocessing Idempotence

*For any* review text string, applying `preprocess_text` twice SHALL produce the same output as applying it once — i.e., `preprocess_text(preprocess_text(text)) == preprocess_text(text)` for all inputs.

**Validates: Requirements 8.1**

---

### Property 2: Sentiment Label Completeness and Correctness

*For any* DataFrame row where `Rating` is a valid integer in [1, 5], the assigned `Sentiment_Label` SHALL be exactly `'Negative'` for ratings 1–2, `'Neutral'` for rating 3, and `'Positive'` for ratings 4–5. No row with a valid rating SHALL have a missing or unexpected label.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

---

### Property 3: Preprocessing Output Alphabet

*For any* input string, the output of `preprocess_text` SHALL contain only lowercase alphabetic characters and single spaces — no digits, punctuation, HTML tags, URLs, or uppercase letters.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

---

### Property 4: Preprocessing Removes Stopwords

*For any* input string, no token in the output of `preprocess_text` (split by whitespace) SHALL appear in the NLTK English stopwords list.

**Validates: Requirements 3.5**

---

### Property 5: EDA Drops Missing Rows

*For any* DataFrame with missing values in `Review` or `Rating`, after `run_eda` the resulting DataFrame SHALL contain no NaN values in either the `Review` or `Rating` column.

**Validates: Requirements 2.3, 2.4**

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| CSV file not found | Print `"Error: File '<path>' not found."` → `sys.exit(1)` |
| CSV unreadable (bad format) | Print `"Error: Could not read CSV — <exception message>"` → `sys.exit(1)` |
| Missing required column | Print `"Error: Missing required column(s): <list>"` → `sys.exit(1)` |
| `output/` directory missing | Auto-created with `os.makedirs(OUTPUT_DIR, exist_ok=True)` |
| Dataset < 50 samples (ML) | Print warning, skip `train_classifier` gracefully |
| NLTK data not downloaded | `nltk.download()` calls at startup guard against missing corpora |

---

## Testing Strategy

### Dual Testing Approach

The pipeline uses both **unit/example-based tests** and **property-based tests** for comprehensive coverage.

### Property-Based Testing

The feature has clear candidates for property-based testing, specifically the `preprocess_text` function (a pure string → string transformation) and the `label_sentiment` logic (a pure integer → string mapping). These are tested using **Hypothesis** (Python's leading PBT library).

**Library**: [`hypothesis`](https://hypothesis.readthedocs.io/)

**Configuration**: Each property test runs a minimum of **100 iterations** (Hypothesis default is 100; can be raised with `@settings(max_examples=200)`).

**Tag format**: Each test is tagged with a comment:
```python
# Feature: flipkart-sentiment-analysis, Property <N>: <property_text>
```

#### Property Tests

| Property | Test Description | Hypothesis Strategy |
|----------|-----------------|---------------------|
| P1: Idempotence | `preprocess_text(preprocess_text(t)) == preprocess_text(t)` | `st.text()` |
| P2: Label correctness | For any rating in [1,5], label is correct | `st.integers(min_value=1, max_value=5)` |
| P3: Output alphabet | Output contains only `[a-z ]` | `st.text()` |
| P4: No stopwords | No token in output is an NLTK stopword | `st.text()` |
| P5: EDA drops NaN | After `run_eda`, no NaN in Review/Rating | Generated DataFrames with injected NaNs |

### Unit / Example-Based Tests

| Test | Description |
|------|-------------|
| `test_load_data_missing_file` | `sys.exit(1)` on missing CSV |
| `test_load_data_missing_column` | `sys.exit(1)` when column absent |
| `test_load_data_success` | Shape and columns correct on valid CSV |
| `test_label_sentiment_boundary` | Rating=3 → Neutral, Rating=4 → Positive, Rating=2 → Negative |
| `test_preprocess_url_removal` | URLs stripped from output |
| `test_preprocess_html_removal` | HTML tags stripped |
| `test_preprocess_lowercase` | All output is lowercase |
| `test_output_dir_created` | `output/` directory created if absent |
| `test_classifier_skipped_small_dataset` | Warning printed and classifier skipped when < 50 rows |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_full_pipeline_smoke` | Run `main()` on a small synthetic CSV; verify output files exist |
