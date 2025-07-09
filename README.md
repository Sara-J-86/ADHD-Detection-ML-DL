![Screenshot 2025-07-09 223633](https://github.com/user-attachments/assets/bae6a2bc-64c4-4eb9-9cd5-f2b20136767b)## ADHD Reddit Post Classification using ML & LSTM

This project explores machine learning and deep learning techniques to classify Reddit posts as ADHD-related or not. Using user-generated Reddit data, the models aim to support early mental health screening and detection.

---

## Objective

To build and compare classical machine learning and deep learning models that can identify whether a Reddit post is related to ADHD.

---

## Dataset

- **ADHD Data:** `ADHD.csv` (Reddit posts from ADHD subreddits)
- **Non-ADHD Data:** `NonADHD.csv` (Posts from general subreddits)
- **Total records after merging & cleaning:** 20,000
- **Combined text:** `title + selftext` for ADHD posts, `body` for non-ADHD posts

---

## Preprocessing

- Lowercasing
- Removal of URLs, special characters, and numbers
- Tokenization
- TF-IDF vectorization (for ML models)
- Tokenizer + Padding (for LSTM)

---

## Models Used

| Model         | Type           | Accuracy |
|---------------|----------------|----------|
| SVM           | Classical ML   | 88%      |
| Random Forest | Ensemble ML    | 95%      |
| XGBoost       | Boosted ML     | 93%      |
| AdaBoost      | Boosted ML     | 80%      |
| LSTM          | Deep Learning  | 96%      |

---

## Techniques Used

- TF-IDF Vectorization for traditional models
- Bidirectional LSTM with Embedding and SpatialDropout
- ROC-AUC, Confusion Matrix, F1-Score, Precision, Recall
- Visualizations using Matplotlib & Seaborn

---

## Model Performance

**Classical Models – Combined ROC Curve**  
![Screenshot 2025-07-09 223638](https://github.com/user-attachments/assets/9c03e8d4-5bf9-4b71-98ae-6dfe049a3044)


**Classical Models – Confusion Matrices**  
![Screenshot 2025-07-09 223633](https://github.com/user-attachments/assets/74b078d9-ccdd-4269-b3f8-9ae0bc871a86)


**LSTM Model – ROC Curve**  
![Screenshot 2025-07-09 223647](https://github.com/user-attachments/assets/91463dc2-43d1-43b9-9055-6bf731c28c09)


**LSTM Model – Confusion Matrix**  
![Screenshot 2025-07-09 223642](https://github.com/user-attachments/assets/870a9ae6-a098-4d84-83da-de58d67154da)


---

## Observations

- LSTM performed best with 96% accuracy and high generalization.
- Random Forest was the top traditional model with 95% accuracy and balanced metrics.
- AdaBoost underperformed due to sensitivity to class imbalance.
- Deep learning models captured semantic patterns better but required more training time and tuning.

---

## Challenges

- Handling noisy, user-generated Reddit text
- Minor overfitting observed in deep learning after a few epochs
- Balancing interpretability with performance (traditional ML is easier to interpret)

---

## Installation

Clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
