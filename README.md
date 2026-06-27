# 🧠 Sentiment Analysis Comparison using RNN, LSTM, and GRU

Project ini merupakan implementasi sederhana **Deep Learning untuk Natural Language Processing (NLP)** menggunakan tiga arsitektur Recurrent Neural Network:

- Simple RNN
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

Dataset yang digunakan adalah **IMDb Movie Reviews Dataset** dengan tujuan melakukan klasifikasi sentimen menjadi **Positive** atau **Negative**.

---

# 📌 Tujuan Project

Project ini bertujuan untuk membandingkan performa tiga model Deep Learning dalam melakukan klasifikasi sentimen teks.

Model yang dibandingkan:

- ✅ Simple RNN
- ✅ LSTM
- ✅ GRU

Evaluasi dilakukan menggunakan beberapa metrik seperti:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# 📂 Struktur Project

```
rnn-lstm-gru-demo/
│
├── dataset/
│   └── IMDB Dataset.csv
│
├── models/
│   ├── rnn.keras
│   ├── lstm.keras
│   └── gru.keras
│
├── output/
│   ├── rnn_confusion_matrix.png
│   ├── lstm_confusion_matrix.png
│   ├── gru_confusion_matrix.png
│   └── comparison.csv
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 📚 Dataset

Dataset yang digunakan:

**IMDb Movie Reviews Dataset**

Jumlah data:

- 50.000 Review
- 25.000 Positive
- 25.000 Negative

Dataset dapat diunduh melalui Kaggle:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Simpan dataset pada folder:

```
dataset/
    IMDB Dataset.csv
```

---

# ⚙️ Library

Install seluruh library

```bash
pip install -r requirements.txt
```

atau

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

---

# ▶️ Menjalankan Program

Pastikan struktur folder sudah benar.

Jalankan:

```bash
python main.py
```

Program akan menjalankan seluruh proses secara otomatis.

---

# 🔄 Workflow

```
IMDb Dataset
      │
      ▼
Load Dataset
      │
      ▼
Text Cleaning
      │
      ▼
Label Encoding
      │
      ▼
Tokenizer
      │
      ▼
Text to Sequence
      │
      ▼
Padding
      │
      ▼
Train/Test Split
      │
      ▼
Embedding Layer
      │
      ▼
Simple RNN
      │
      ├───────────────┐
      ▼               │
LSTM                  │
      │               │
      ▼               │
GRU                   │
      └───────────────┘
              │
              ▼
Evaluation
              │
              ▼
Visualization
```

---

# 🧠 Arsitektur Model

## 1. Simple RNN

```
Embedding
      │
SimpleRNN (64)
      │
Dropout
      │
Dense (32)
      │
Dense (1)
```

---

## 2. LSTM

```
Embedding
      │
LSTM (64)
      │
Dropout
      │
Dense (32)
      │
Dense (1)
```

---

## 3. GRU

```
Embedding
      │
GRU (64)
      │
Dropout
      │
Dense (32)
      │
Dense (1)
```

---

# 📊 Evaluasi

Model dievaluasi menggunakan:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Contoh hasil:

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|---------:|----------:|--------:|---------:|
| Simple RNN | 84.5% | 84.2% | 84.8% | 84.5% |
| LSTM | 89.4% | 89.3% | 89.5% | 89.4% |
| GRU | 88.7% | 88.6% | 88.8% | 88.7% |

> Nilai di atas hanya contoh. Hasil aktual bergantung pada proses training.

---

# 📈 Output

Project akan menghasilkan:

## Model

```
models/

rnn.keras

lstm.keras

gru.keras
```

## Confusion Matrix

```
output/

rnn_confusion_matrix.png

lstm_confusion_matrix.png

gru_confusion_matrix.png
```

---

# 📊 Contoh Confusion Matrix

```
                 Predicted

              Negative Positive

Actual Negative   4300      700

Actual Positive    450     4550
```

---

# 📌 Hyperparameter

| Parameter | Nilai |
|-----------|-------:|
| Vocabulary Size | 10000 |
| Sequence Length | 200 |
| Embedding Dimension | 128 |
| Batch Size | 64 |
| Epoch | 5 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |

---

# 📦 Requirements

```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

# 🚀 Pengembangan Selanjutnya

Project ini dapat dikembangkan menjadi:

- Bidirectional LSTM
- Bidirectional GRU
- Attention Mechanism
- CNN + LSTM
- Word2Vec Embedding
- GloVe Embedding
- FastText Embedding
- BERT
- RoBERTa
- Streamlit Dashboard

---

# 👨‍💻 Author

**Arif**

Master's Student in Information Technology

Fokus:

- Artificial Intelligence
- Deep Learning
- Natural Language Processing
- Data Science

---

# 📖 Referensi

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.

2. Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder.

3. TensorFlow Documentation

https://www.tensorflow.org/

4. IMDb Movie Reviews Dataset

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

# ⭐ Jika project ini bermanfaat

Silakan berikan ⭐ pada repository ini.
