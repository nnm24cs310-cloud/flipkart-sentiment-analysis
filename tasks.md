# Implementation Tasks: Flipkart Customer Sentiment Analysis

## Task List

- [x] 1. Project scaffold and configuration constants
  - [x] 1.1 Create `project.py` in the workspace root with a module-level docstring describing the pipeline
  - [x] 1.2 Add configuration constants at the top: `CSV_PATH`, `ENABLE_ML`, `RANDOM_SEED`, `MAX_TFIDF_FEATURES`, `OUTPUT_DIR`
  - [x] 1.3 Add NLTK corpus download calls (`stopwords`, `wordnet`, `omw-1.4`) with `quiet=True` so they run once at import time

- [x] 2. Implement `load_data(path: str) -> pd.DataFrame`
  - [x] 2.1 Use `pd.read_csv(path)` inside a try/except block; on `FileNotFoundError` print a descriptive error and call `sys.exit(1)`; on any other exception print the exception message and call `sys.exit(1)`
  - [x] 2.2 Validate that the loaded DataFrame contains all three required columns (`Product_name`, `Review`, `Rating`); if any are missing, print the missing column names and call `sys.exit(1)`
  - [x] 2.3 Print the DataFrame shape and the first five rows on successful load
  - [x] 2.4 Return the loaded DataFrame

- [x] 3. Implement `run_eda(df: pd.DataFrame) -> pd.DataFrame`
  - [x] 3.1 Print summary statistics for the `Rating` column using `df['Rating'].describe()`
  - [x] 3.2 Print missing value counts for all columns using `df.isnull().sum()`
  - [x] 3.3 Drop rows where `Review` is NaN; print the number of rows removed before dropping
  - [x] 3.4 Drop rows where `Rating` is NaN; print the number of rows removed before dropping
  - [x] 3.5 Print the distribution of `Rating` values using `value_counts().sort_index()`
  - [x] 3.6 Print the top 10 most reviewed products using `df['Product_name'].value_counts().head(10)`
  - [x] 3.7 Return the cleaned DataFrame

- [x] 4. Implement `preprocess_text(text: str) -> str`
  - [x] 4.1 Convert the input string to lowercase
  - [x] 4.2 Remove URLs using a regex pattern that matches `http://`, `https://`, and `www.` prefixed strings
  - [x] 4.3 Remove HTML tags using a regex pattern `<[^>]+>`
  - [x] 4.4 Remove all characters that are not lowercase alphabetic or whitespace using `re.sub(r'[^a-z\s]', ' ', text)`
  - [x] 4.5 Tokenize the cleaned string by splitting on whitespace
  - [x] 4.6 Remove NLTK English stopwords from the token list
  - [x] 4.7 Apply `WordNetLemmatizer().lemmatize()` to each remaining token
  - [x] 4.8 Join the tokens back into a single string and return it
  - [x] 4.9 Handle the edge case where `text` is not a string (e.g., NaN/float) by converting to empty string before processing

- [x] 5. Implement `apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame`
  - [x] 5.1 Apply `preprocess_text` to every value in `df['Review']` using `.apply()` and store the result in `df['Cleaned_Review']`
  - [x] 5.2 Print the first five rows showing `Review` and `Cleaned_Review` columns side-by-side
  - [x] 5.3 Return the modified DataFrame

- [x] 6. Implement `label_sentiment(df: pd.DataFrame) -> pd.DataFrame`
  - [x] 6.1 Use `np.select` (or equivalent vectorized approach) to map `Rating` values to `Sentiment_Label`: 1–2 → `'Negative'`, 3 → `'Neutral'`, 4–5 → `'Positive'`
  - [x] 6.2 Store the result in `df['Sentiment_Label']`
  - [x] 6.3 Print the value counts for `Sentiment_Label`
  - [x] 6.4 Return the modified DataFrame

- [x] 7. Implement `generate_visualizations(df: pd.DataFrame) -> None`
  - [x] 7.1 Create the `output/` directory using `os.makedirs(OUTPUT_DIR, exist_ok=True)` before saving any file
  - [x] 7.2 Generate and save `rating_distribution.png`: a bar chart of `Rating` value counts with title, x-label (`Rating`), y-label (`Count`), and appropriate tick labels
  - [x] 7.3 Generate and save `sentiment_distribution.png`: a pie chart of `Sentiment_Label` proportions with title and legend showing label names and percentages
  - [x] 7.4 Generate and save `top_products.png`: a horizontal bar chart of the top 10 most reviewed products with title, x-label (`Review Count`), and y-label (`Product`)
  - [x] 7.5 Generate and save `wordcloud_positive.png`: a word cloud built from the concatenated `Cleaned_Review` text of all `Positive` rows; include a title
  - [x] 7.6 Generate and save `wordcloud_negative.png`: a word cloud built from the concatenated `Cleaned_Review` text of all `Negative` rows; include a title
  - [x] 7.7 Close each matplotlib figure after saving to free memory (`plt.close()`)

- [x] 8. Implement `correlation_analysis(df: pd.DataFrame) -> None`
  - [x] 8.1 Compute and print the average `Rating` for each `Sentiment_Label` group using `groupby`
  - [x] 8.2 Generate a box plot of `Rating` distribution grouped by `Sentiment_Label` using seaborn or matplotlib with title, x-label (`Sentiment Label`), and y-label (`Rating`)
  - [x] 8.3 Save the box plot as `output/rating_by_sentiment_boxplot.png`
  - [x] 8.4 Close the figure after saving

- [x] 9. Implement `train_classifier(df: pd.DataFrame) -> None`
  - [x] 9.1 Return immediately (with a printed warning) if `len(df) < 50`
  - [x] 9.2 Vectorize `df['Cleaned_Review']` using `TfidfVectorizer(max_features=MAX_TFIDF_FEATURES)` and fit on the full corpus
  - [x] 9.3 Split the vectorized features and `Sentiment_Label` target into 80/20 train/test sets using `train_test_split` with `random_state=RANDOM_SEED`
  - [x] 9.4 Train a `LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)` model on the training set
  - [x] 9.5 Generate predictions on the test set and print the full `classification_report` (accuracy, precision, recall, F1 per class)
  - [x] 9.6 Compute the confusion matrix and generate a seaborn heatmap with annotations, title (`Confusion Matrix`), x-label (`Predicted`), y-label (`Actual`), and class labels
  - [x] 9.7 Save the confusion matrix heatmap as `output/confusion_matrix.png` and close the figure
  - [x] 9.8 Print the top 10 most influential TF-IDF feature names for each sentiment class by examining the model's `coef_` attribute

- [x] 10. Implement `main()` and wire the pipeline
  - [x] 10.1 Call `load_data(CSV_PATH)` and store the result
  - [x] 10.2 Call `run_eda(df)` and store the cleaned DataFrame
  - [x] 10.3 Call `apply_preprocessing(df)` and store the result
  - [x] 10.4 Call `label_sentiment(df)` and store the result
  - [x] 10.5 Call `generate_visualizations(df)`
  - [x] 10.6 Call `correlation_analysis(df)`
  - [x] 10.7 If `ENABLE_ML` is `True`, call `train_classifier(df)`
  - [x] 10.8 Add `if __name__ == '__main__': main()` guard at the bottom of the file

- [x] 11. Write property-based tests using Hypothesis
  - [x] 11.1 Create `test_project.py` in the workspace root and import `preprocess_text`, `label_sentiment`, and `run_eda` from `project`
  - [x] 11.2 Write property test P1 (Idempotence): use `@given(st.text())` to verify `preprocess_text(preprocess_text(t)) == preprocess_text(t)` for all inputs
    - Tag: `# Feature: flipkart-sentiment-analysis, Property 1: Preprocessing Idempotence`
  - [x] 11.3 Write property test P2 (Label correctness): use `@given(st.integers(min_value=1, max_value=5))` to verify the correct `Sentiment_Label` is assigned for every valid rating
    - Tag: `# Feature: flipkart-sentiment-analysis, Property 2: Sentiment Label Completeness and Correctness`
  - [x] 11.4 Write property test P3 (Output alphabet): use `@given(st.text())` to verify the output of `preprocess_text` matches `^[a-z ]*$`
    - Tag: `# Feature: flipkart-sentiment-analysis, Property 3: Preprocessing Output Alphabet`
  - [x] 11.5 Write property test P4 (No stopwords): use `@given(st.text())` to verify no token in the output of `preprocess_text` appears in the NLTK English stopwords set
    - Tag: `# Feature: flipkart-sentiment-analysis, Property 4: Preprocessing Removes Stopwords`
  - [x] 11.6 Write property test P5 (EDA drops NaN): use `@given` with a strategy that builds DataFrames with random NaN injection in `Review` and `Rating` columns, then verify `run_eda` returns a DataFrame with no NaN in those columns
    - Tag: `# Feature: flipkart-sentiment-analysis, Property 5: EDA Drops Missing Rows`
  - [x] 11.7 Annotate all property tests with `@settings(max_examples=100)` to ensure minimum 100 iterations

- [-] 12. Write unit and example-based tests
  - [ ] 12.1 Write `test_load_data_missing_file`: assert `SystemExit` is raised when a non-existent path is passed to `load_data`
  - [ ] 12.2 Write `test_load_data_missing_column`: create a temporary CSV without the `Rating` column and assert `SystemExit` is raised
  - [ ] 12.3 Write `test_load_data_success`: create a valid temporary CSV and assert the returned DataFrame has the correct shape and columns
  - [ ] 12.4 Write `test_label_sentiment_boundaries`: assert Rating=1 → `Negative`, Rating=2 → `Negative`, Rating=3 → `Neutral`, Rating=4 → `Positive`, Rating=5 → `Positive`
  - [ ] 12.5 Write `test_preprocess_url_removal`: assert URLs are absent from `preprocess_text` output
  - [ ] 12.6 Write `test_preprocess_html_removal`: assert HTML tags are absent from `preprocess_text` output
  - [ ] 12.7 Write `test_preprocess_lowercase`: assert output of `preprocess_text` contains no uppercase characters
  - [ ] 12.8 Write `test_output_dir_created`: assert the `output/` directory is created when `generate_visualizations` is called and the directory does not exist
  - [ ] 12.9 Write `test_classifier_skipped_small_dataset`: create a DataFrame with fewer than 50 rows and assert `train_classifier` prints a warning and returns without raising an exception

- [ ] 13. Verify the full pipeline runs end-to-end
  - [ ] 13.1 Run `python project.py` with a sample CSV to confirm all pipeline stages execute without errors
  - [ ] 13.2 Confirm all expected PNG files are present in the `output/` directory after execution
  - [ ] 13.3 Run `pytest test_project.py -v` and confirm all unit and property tests pass
