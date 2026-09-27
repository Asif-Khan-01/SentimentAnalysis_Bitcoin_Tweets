# Bitcoin Tweet Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLP-NLTK%20%7C%20VADER-2E8B57)

A decision support pipeline that classifies the sentiment of ~50,000 Bitcoin tweets and tests whether daily Twitter sentiment helps explain Bitcoin price direction.

**Team project at Blekinge Institute of Technology (BTH).** Team: Muhammad Asif Khan and Syeda Sara Afzaal.

![Model comparison](model_comparison_dashboard.png)

## Results

| Task | Model | Accuracy | Notes |
|---|---|---|---|
| Tweet sentiment (positive vs. negative/neutral) | **Bidirectional LSTM** | **95.8%** | Precision and recall 0.96 |
| Tweet sentiment (positive vs. negative/neutral) | Logistic Regression + TF-IDF | 92.0% | Fast, interpretable baseline |
| Daily price direction (up/down) | Random Forest | 42.9% | Only 14 test days, see limitations |

**Key finding:** For daily price direction, price volatility (41%) and trading volume (34%) carried most of the feature importance, while average sentiment, tweet count and engagement each contributed about 7%. On this data, tweet sentiment on its own was a weak signal for price movement.

## Pipeline

```
Tweets ─► clean text ─► VADER labels ─► LR / BiLSTM sentiment classifiers
                            │
                            └─► daily sentiment + BTC-USD prices ─► Random Forest
```

1. **Data:** 49,921 tweets sampled from the [Kaggle Bitcoin Tweets dataset](https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329), covering 5 February to 12 April 2021 (66 days), plus daily BTC-USD prices from Yahoo Finance.
2. **Cleaning:** URLs, mentions, hashtags, special characters and stopwords removed; text lower-cased.
3. **Labelling:** VADER compound score, with positive at 0.05 or above, negative at -0.05 or below, and neutral in between. Result: 44% positive, 11% negative, 45% neutral.
4. **Sentiment models:** Logistic Regression on 5,000 TF-IDF features (1 to 2-grams), and a BiLSTM (Embedding 128, Bi-LSTM 64, Bi-LSTM 32, Dense 64, dropout 0.5, 10 epochs).
5. **Price model:** Random Forest (100 trees, max depth 10) on daily sentiment, tweet volume, engagement, price range and trading volume, with a time-ordered 80/20 split.

## Limitations

- The sentiment models learn to reproduce VADER labels, so their accuracy measures agreement with VADER rather than with human-labelled sentiment.
- The price model has only 66 days of data (14 test days), and it uses same-day price range and volume, so it explains a day's direction rather than forecasting the next day.
- Engagement fields (likes, retweets) were missing in the source data.

Natural next steps: human-labelled sentiment, transformer models (for example FinBERT), a longer time range and true next-day forecasting with lagged features.

## Project structure

| Step | Script | Output |
|---|---|---|
| 1 | `prepare_Kdata.py` | Prepares the Kaggle tweet file |
| 2 | `preprocess_data.py` | `tweets_cleaned.csv`, `daily_sentiment.csv` |
| 3 | `collect_prices.py` | `bitcoin_prices.csv` (Yahoo Finance) |
| 4 | `fix_merge.py` | `final_dataset.csv` (daily sentiment + prices) |
| 5 | `logistic_regression.py` | `model1_logistic_regression.pkl`, confusion matrix |
| 6 | `random_forest.py` | `model2_random_forest.pkl`, feature importance |
| 7 | `lstm_model.py` | `model3_lstm.h5`, tokenizer, training history |
| 8 | `compare_models.py` | `model_comparison_dashboard.png` |

Other files: `dashboard.html` (interactive results dashboard), `check_data.py` (data sanity checks), `collect_tweets.py` (synthetic sample generator used during development, not used for the final results).

## Getting started

```bash
git clone https://github.com/Asif-Khan-01/SentimentAnalysis_Bitcoin_Tweets.git
cd SentimentAnalysis_Bitcoin_Tweets
pip install -r requirements.txt
```

The trained models, processed data and charts are included, so results can be explored straight away by opening `dashboard.html`.

To rerun everything, download `BitcoinTweets.csv` (about 2 GB) from Kaggle into the project folder and run the scripts in the order above.

## Visuals

| LSTM training | LSTM confusion matrix | Random Forest feature importance |
|---|---|---|
| ![Training](model3_training_history.png) | ![Confusion](model3_confusion_matrix.png) | ![Features](model2_feature_importance.png) |

## Team

- **Muhammad Asif Khan** · [GitHub](https://github.com/Asif-Khan-01) · [LinkedIn](https://www.linkedin.com/in/muhammad-asif-khan-3076801bb)
- **Syeda Sara Afzaal** · [GitHub](https://github.com/sawraw404) · [LinkedIn](https://www.linkedin.com/in/sara-afzaal-0b6690297/)

Original repository: [sawraw404/SentimentAnalysis_Bitcoin_Tweets](https://github.com/sawraw404/SentimentAnalysis_Bitcoin_Tweets)

## License

MIT License. See [LICENSE](LICENSE).
