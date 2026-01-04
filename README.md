# 🎵 Lyric Search & Artist Prediction using Machine Learning

## 📌 Project Overview

This project implements a **Lyric Search and Artist Prediction system** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. Given a song lyric or text input, the model predicts the **most likely artist** associated with the lyrics.

The solution uses **TF-IDF vectorization** for text representation and a **TensorFlow neural network** for multi-class classification.

---

## 🎯 Objectives

* Preprocess song lyrics using NLP techniques
* Convert text data into numerical features using TF-IDF
* Train a deep learning model to classify lyrics by artist
* Predict the artist for unseen lyric inputs

---

## 🛠️ Technologies Used

* **Python 3.x**
* **TensorFlow**
* **Scikit-learn**
* **Pandas & NumPy**
* **NLTK**
* **Jupyter Notebook**

---

## 📂 Project Structure

```
📁 Lyric-Search-ML
│
├── lyric_search.ipynb     # Main notebook (model training & prediction)
├── README.md              # Project documentation
└── requirements.txt       # (Optional) Dependencies
```

---

## 🔍 Workflow Explanation

1. **Data Loading** – Load lyric dataset containing lyrics and artist labels
2. **Text Preprocessing**

   * Lowercasing
   * Removing punctuation and numbers
   * Tokenization
   * Stopword removal
3. **Feature Engineering**

   * TF-IDF Vectorization
4. **Label Encoding**

   * Convert artist names into numerical labels
5. **Model Training**

   * Neural network built using TensorFlow
6. **Prediction**

   * Predict artist based on new lyric input

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/lyric-search-ml.git
cd lyric-search-ml
```

---

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # For Linux / Mac
venv\Scripts\activate         # For Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install tensorflow
pip install pandas numpy scikit-learn nltk
```

OR (if using `requirements.txt`):

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download NLTK Resources

Run the following in Python:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## ▶️ Execution (How to Run)

### 1️⃣ Start Jupyter Notebook

```bash
jupyter notebook
```

### 2️⃣ Open the Notebook

Open:

```
lyric_search.ipynb
```

### 3️⃣ Run All Cells

* Click **Kernel → Restart & Run All**
* Model will train automatically

---

## 🧪 Example Usage

```python
predict_song_artist("I'm walking alone down this empty road")
```

**Output:**

```
'Artist Name'
```

---

## 📊 Model Details

* **Vectorization:** TF-IDF
* **Model Type:** Multi-class Neural Network
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam
* **Evaluation Metric:** Accuracy

---

## 🚀 Future Enhancements

* Use **LSTM / Transformer models**
* Add **Top-K artist predictions**
* Deploy as a **web application**
* Improve dataset size for higher accuracy

---

## 🧠 Learning Outcomes

* Hands-on NLP preprocessing
* Practical TF-IDF implementation
* Deep learning text classification
* End-to-end ML pipeline development

---

## 👨‍💻 Author-Anustup Das

