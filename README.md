

````markdown
# 🛍️ Flipkart Customer Sentiment Analyzer

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-blueviolet)](https://www.nltk.org/)

An interactive web application built with Streamlit that uses a pre-trained **Linear SVM model (89.5% accuracy)** to perform real-time sentiment analysis on customer reviews.

This project analyzes user-provided text to classify sentiment as **Positive**, **Negative**, or **Neutral**, providing a clear "Happy / Not Happy" verdict and visual analytics.

---

## 🚀 Live Application



*This README is for the deployed Streamlit application. The original analysis notebook used to train the model on 200,000+ Flipkart reviews can be found in `flipkart_sentiment_analysis.ipynb`.*

## ✨ Key Features

* **Real-Time Analysis:** Instantly classifies sentiment for one or more reviews.
* **Simple Interface:** Users paste reviews (one per line) into a text box.
* **Clear Verdict:** Provides a high-level "Happy," "Not Happy," or "Mixed" verdict.
* **Visual Dashboard:** Displays a bar chart showing the breakdown of sentiments.
* **Detailed Results:** Shows a data table with the original review and its predicted sentiment.

## 🛠️ Technical Stack

* **Backend & ML:** Python, Scikit-learn, Pandas, NLTK
* **Frontend Web App:** Streamlit
* **Model:** Linear SVM (Support Vector Machine)
* **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
* **NLP Pipeline:** NLTK (Stopword removal, Snowball Stemmer)
* **Deployment:** Streamlit Cloud (or local)

---

## ⚙️ How It Works: The ML Pipeline

The model at the heart of this app was trained on over 200,000 Flipkart product reviews.

1.  **Text Cleaning (NLTK):**
    * Converts text to lowercase.
    * Removes all punctuation, URLs, and HTML tags.
    * Removes common English **stopwords** (e.g., "the", "is", "a").
    * Performs **stemming** (e.g., "running" -> "run") using `SnowballStemmer`.

2.  **Feature Extraction (TF-IDF):**
    * The cleaned text is converted into a numerical matrix using **TF-IDF**.
    * This technique scores words based on their frequency in a single review vs. their rarity across all reviews.
    * The model was trained on **5,000** of the most important word features (including 1 and 2-word phrases).

3.  **Model Training (Linear SVM):**
    * Several models were tested (Logistic Regression, Naive Bayes), but **Linear SVM** provided the best performance, achieving **89.5% accuracy** on the test dataset.
    * The trained model, TF-IDF vectorizer, and label encoder are all saved as `.pkl` files and loaded by the Streamlit app to make live predictions.



---

## 🚦 Installation & Usage

You can run this application on your local machine in just a few steps.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/flipkart-sentiment.git](https://github.com/your-username/flipkart-sentiment.git)
cd flipkart-sentiment
````

### 2\. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Requirements

Make sure you have your 3 model files (`sentiment_model.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`) in the root folder.

```bash
pip install -r requirements.txt
```

### 4\. Run the Streamlit App

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501` to display the app.

-----

## 📁 File Structure

```
.
├── sentiment_model.pkl             # <-- The trained SVM model
├── tfidf_vectorizer.pkl          # <-- The fitted TF-IDF vectorizer
├── label_encoder.pkl             # <-- The fitted LabelEncoder
|
├── app.py                          # The main Streamlit application code
├── requirements.txt                # Python libraries needed to run the app
|
├── flipkart_sentiment_analysis.ipynb # (Optional) Original notebook for training
├── Dataset-SA.csv                  # (Optional) The raw dataset
└── README.md                       # You are here!
```

```
```
