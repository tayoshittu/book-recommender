# 📚 Book Recommender

An end-to-end semantic book recommendation system that:

* **Cleans & explores** a 7K book metadata dataset.
* **Indexes** book descriptions into a vector database using OpenAI embeddings and Chroma.
* **Classifies** books into broad categories with zero-shot learning.
* **Analyzes** emotional tone via a transformer-based sentiment pipeline.
* **Serves** an interactive Gradio dashboard for personalized recommendations.

---

## 📂 Repository Structure

```text
├── data-exploration.ipynb      # Cleaning, EDA, and feature engineering (missing values, correlations)
├── vector-search.ipynb         # Building and querying a Chroma vector index of book descriptions
├── text-classification.ipynb   # Zero-shot classification of books into Fiction/Nonfiction
├── sentiment-analysis.ipynb    # Emotion scoring of descriptions using a distilRoBERTa model
├── gradio-dashboard.py         # Gradio app combining semantic search + emotion/category filters
├── books_cleaned.csv           # Cleaned CSV after data exploration
├── tagged_description.txt      # ISBN-tagged descriptions for vector indexing
├── books_with_categories.csv   # Output after mapping and predicting categories
├── books_with_emotions.csv     # Output after computing emotion scores
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and instructions
```

---

## 🚀 Features

1. **Data Exploration & Cleaning**

   * Identifies and visualizes missing values.
   * Filters descriptions with fewer than 25 words.
   * Creates unified `title_and_subtitle` and `tagged_description` fields.

2. **Vector Search**

   * Splits cleaned descriptions into document chunks.
   * Builds a Chroma vector store with OpenAIEmbeddings.
   * Retrieves top-k semantically similar books by query.

3. **Zero-Shot Classification**

   * Maps detailed categories into `simple_categories` (e.g., Children’s Fiction, Nonfiction).
   * Uses Hugging Face’s BART-MNLI pipeline to classify books into `Fiction` or `Nonfiction`.

4. **Sentiment Analysis**

   * Applies a distilRoBERTa emotion classifier (`j-hartmann/emotion-english-distilroberta-base`).
   * Aggregates max emotion scores (anger, joy, sadness, etc.) per book.

5. **Interactive Dashboard**

   * Gradio Blocks interface with:

     * Text search input.
     * Category & emotion-tone filters.
     * Gallery of recommended book covers and truncated descriptions.

---

## ⚙️ Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/<your-username>/book-recommender.git
   cd book-recommender
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   # On macOS/Linux
   export OPENAI_API_KEY="your_openai_api_key"
   # On Windows (PowerShell)
   $env:OPENAI_API_KEY = "your_openai_api_key"
   ```

---

## 🏃‍♂️ Usage

### 1. Data Preparation & Exploration

* Run **data-exploration.ipynb** to clean the raw `books.csv` and output `books_cleaned.csv` and `tagged_description.txt`.

### 2. Build Vector Index

* Run **vector-search.ipynb** to ingest `books_cleaned.csv`, split descriptions, and create a Chroma index (`db_books`).

### 3. Text Classification

* Run **text-classification.ipynb** to map categories and predict missing ones via zero-shot classification.

### 4. Sentiment Analysis

* Run **sentiment-analysis.ipynb** to compute emotion scores and save `books_with_emotions.csv`.

### 5. Launch Dashboard

```bash
python gradio-dashboard.py
```

* Navigate to `http://localhost:7860` in your browser.
* Enter a book description, select a category & emotion tone, and view recommended covers.

---

## 📈 Dependencies

* **Data & Visualization:** `pandas`, `numpy`, `seaborn`, `matplotlib`
* **NLP & Vectors:** `langchain_community`, `langchain_openai`, `langchain_text_splitters`, `langchain_chroma`
* **Deep Learning:** `openai`, `transformers`
* **Dashboard:** `gradio`
* **Utilities:** `python-dotenv`, `kagglehub`, `tqdm`

See [requirements.txt](requirements.txt) for exact versions.

---

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests for enhancements or bug fixes.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
