# Book Review Quality Classification using Snorkel

## Dataset

**Source:** [Amazon Books Reviews — Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)

The dataset consists of two CSV files containing Amazon book reviews and book metadata.

### `Books_rating.csv` (~2.86 GB, ~3M reviews)

| Column | Description |
|---|---|
| `Id` | Book ISBN/ASIN identifier |
| `Title` | Book title |
| `Price` | Book price (frequently empty) |
| `User_id` | Amazon user ID |
| `profileName` | Reviewer display name (sometimes empty) |
| `review/helpfulness` | String fraction like `"7/11"` — helpful votes out of total votes |
| `review/score` | Rating from 1.0 to 5.0 |
| `review/time` | Unix timestamp |
| `review/summary` | Short headline written by reviewer |
| `review/text` | Full review body |

### `books_data.csv` (~181 MB)

| Column | Description |
|---|---|
| `Title` | Book title (join key with ratings) |
| `description` | Publisher/book description |
| `authors` | Python list-style string, e.g. `['Philip Nel']` |
| `image` | Google Books cover image URL |
| `previewLink` | Google Books preview URL |
| `publisher` | Publisher name |
| `publishedDate` | Publication date |
| `infoLink` | Google Books info URL |
| `categories` | Python list-style string, e.g. `['Fiction']` |
| `ratingsCount` | Aggregate rating count |

### Known Data Issues

- **Size:** The ratings file is 2.86 GB. Loading it fully into memory is impractical for interactive work, so we sample a fraction during processing.
- **Missing values:** `Price`, `profileName`, and `User_id` are frequently empty. Some reviews have no `review/text` at all.
- **HTML entities in text:** Review text contains unescaped HTML like `&quot;`, `&amp;`, `&lt;` instead of actual characters.
- **Missing spaces:** Paragraph breaks were stripped during data collection, so sentences run together: `"...paintings by Olivia.If you're looking..."`.
- **Helpfulness is a string:** The `review/helpfulness` column stores fractions as strings (`"7/11"`) rather than numeric values.
- **Timestamps are unix integers:** `review/time` contains raw epoch seconds, not readable dates.
- **Metadata columns are string-encoded lists:** `authors` and `categories` in `books_data.csv` are stored as string representations of Python lists, not actual lists.
- **Duplicate reviews exist:** Some books have duplicate or near-duplicate reviews, including evidence of review manipulation (fake reviews written in identical style).
- **Review types are mixed:** The dataset contains transaction reviews ("shipped fast, good condition"), gift reviews ("bought this for my daughter"), and assigned-reading reviews ("needed it for class") alongside genuine book reviews. The star rating does not distinguish between these.

---

## Goal

### The Problem

Every review has a star rating (1–5), but the rating only measures *satisfaction* — not whether the review is actually useful. A 5-star review that says "arrived on time, great condition" tells a potential reader nothing about the book. A 1-star review with detailed criticism of the plot and writing is far more valuable. The star rating cannot distinguish between these.

### What We Build

A classifier that labels each review as:

- **HIGH_QUALITY:** A substantive review that discusses the book's content, writing, themes, or value
- **LOW_QUALITY:** A transaction review, no-content review, troll review, or off-topic review

### Weak Supervision with Snorkel

Since we don't have hand-labeled ground truth for "review quality," we use **weak supervision** through [Snorkel](https://www.snorkel.org/) to programmatically generate training labels.

The process works as follows:

1. **Labeling Functions (LFs):** We write multiple heuristic functions that each look at a review and vote HIGH_QUALITY, LOW_QUALITY, or ABSTAIN. Each LF captures one signal — keyword patterns, review length, helpfulness votes, sentiment scores, etc. Individual LFs are noisy and incomplete, and that's expected.

2. **Label Model:** Snorkel's `LabelModel` takes the votes from all LFs and learns how accurate each one is by analyzing their agreements and disagreements. It then combines their outputs into a single probabilistic label per review — without needing any hand-labeled data.

3. **Discriminative Classifier:** We train a standard classifier (Logistic Regression) on the Snorkel-generated labels. This classifier generalizes beyond what the LFs cover and can predict quality for any new review using only the text.

4. **Data Slicing:** We define slicing functions that identify critical subsets of the data (short reviews, reviews with no helpfulness votes, fiction vs. non-fiction) and monitor how the model performs on each slice. This catches cases where overall accuracy looks good but specific subsets fail.

---

## Process

The notebook `book_review_quality_snorkel.ipynb` follows these steps:

### 1. Data Loading & Cleaning

- Chunk-loads the 2.86 GB ratings file and samples 0.5% for manageable processing
- Drops reviews with no text, fills missing profile names
- Decodes HTML entities in review text and summaries
- Fixes missing spaces between sentences
- Parses the helpfulness string fraction into numeric columns (`helpful`, `total_votes`, `help_ratio`)
- Converts unix timestamps to datetime
- Precomputes word counts for review text and summaries
- Removes duplicate reviews (same book + same text)
- Loads book metadata, parses string-encoded category lists, and joins on `Title`
- Splits into train (85%) and test (15%) sets

### 2. Writing Labeling Functions (18 LFs)

- **Helpfulness-based (2):** Reviews with high/low helpfulness ratios as distant supervision
- **Transaction detectors (2):** Short reviews about shipping/condition/gifts rather than the book
- **Keyword LFs (4):** Literary analysis terms, academic language, book comparisons, meta-review language
- **Heuristic LFs (3):** Very short reviews, substantial-length moderate-score reviews, summary-equals-text detection
- **Regex LFs (3):** Recommendation phrases with reasoning, star-count endings, spoiler warnings
- **TextBlob sentiment LFs (3):** Balanced subjectivity as quality signal, extreme-polarity-plus-short as low quality, score-text sentiment mismatch
- **Category-enriched LF (1):** Long reviews on academic/non-fiction books

### 3. Apply LFs and Analyze

- Applies all 18 LFs to produce a label matrix using `PandasLFApplier`
- Generates `LFAnalysis` summary showing each LF's polarity, coverage, overlaps, and conflicts
- Visualizes coverage distribution and spot-checks labeled examples

### 4. Label Model

- Fits Snorkel's `LabelModel` on the label matrix (500 epochs)
- Produces probabilistic labels and visualizes confidence distribution
- Filters out unlabeled data points (reviews no LF voted on)

### 5. Train Classifier

- Featurizes review text using bag-of-bigrams (`CountVectorizer`)
- Converts probabilistic labels to hard labels using `probs_to_preds`
- Trains a `LogisticRegression` classifier on the weakly-labeled data
- Evaluates on test set using proxy ground truth derived from helpfulness and length signals

### 6. Data Slicing

- Defines 5 slicing functions: short reviews, very long reviews, no-votes reviews, extreme-score reviews, fiction category
- Applies slicing functions to test set and reports slice sizes

### 7. End-to-End Results

- Shows highest-confidence HIGH_QUALITY and LOW_QUALITY predictions with full context
- Highlights the key outputs: 5-star reviews correctly identified as LOW_QUALITY and 1-star reviews correctly identified as HIGH_QUALITY
- Prints pipeline summary statistics

---

## Assumptions

1. **Proxy Labels as Ground Truth:** Due to the lack of actual hand-labeled ground truth for review quality, we created proxy labels derived from helpfulness ratios and review length, and assumed them to be ground truth. This was done in order to learn the process of how weak supervision helps in training a classifier that could predict a true label at great accuracy. In a production setting, a subset of reviews would be manually labeled by human annotators to serve as real ground truth.

2. **Evaluation Limitations:** Since we lack true labels, the accuracy and evaluation metrics reported in the notebook won't certify a satisfactory trained model. This further restricts fine-tuning the model — hyperparameter optimization, threshold tuning, and model selection all require reliable evaluation against real labels. Because of this, the results actually labelled as HIGH_QUALITY or LOW_QUALITY won't make much sense as final outputs for this lab experiment. They serve to demonstrate the pipeline mechanics, not to deliver production-grade predictions.

3. **Test Set as New Dataset:** The test set is assumed to be a different dataset, just to present how a new unseen dataset would be labelled based on the trained classifier. In practice, the classifier trained via weak supervision would be deployed to label incoming reviews that no labeling function has ever seen.

---

## Setup

### Clone the repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

### Download the dataset

Download both CSV files from the [Kaggle dataset page](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) and place them in the project root directory:

```
├── Books_rating.csv
├── books_data.csv
├── book_review_quality_snorkel.ipynb
└── README.md
```

### Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib snorkel textblob jupyter
```

### Launch Jupyter Notebook

```bash
jupyter notebook book_review_quality_snorkel.ipynb
```

Run the cells sequentially. The first cell that loads data will take a minute or two due to chunk-reading the 2.86 GB file.
