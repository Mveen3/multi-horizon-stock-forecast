# Nifty50 Trade Decision Support: News-Sentiment to TFT Forecasting Pipeline

## What This Project Does

This repository builds an end-to-end forecasting workflow for Indian equities:

1. Scrape market news + stock data.
2. Clean and normalize articles.
3. Split article impact into `direct`, `sectoral`, and `global`.
4. Score sentiment with FinBERT.
5. Merge sentiment with OHLCV and engineer features.
6. Train/evaluate TFT across multiple windows.
7. Generate evaluation visualizations.


---

## Public Dataset and Model Links

- Kaggle Dataset: `https://www.kaggle.com/datasets/mveen3/6-year-indian-stock-market-dataset-news-and-ticker`
- Kaggle Model: `https://www.kaggle.com/models/mveen3/tft-model-for-stock-price-prediction`
- Hugging Face Dataset: `https://huggingface.co/datasets/mveen3/Six_Year_Indian_Stock_Market_Dataset-News_and_Ticker`
- Hugging Face Model: `https://huggingface.co/mveen3/TFT_Model_For_Stock_Price_Prediction`

---

## Repository Structure (Current View)

```text
root/
|-- src/
|   |-- 0_preprocess_news.py
|   |-- 1a_openai_news_segregation.py
|   |-- 1b_qwen_news_segregation.py
|   |-- 2_finbert_sentiment.py
|   |-- 3_feature_engineering.py
|   |-- 4_tft_hpt_train_test.py
|   |-- 5_tft_visualize.py
|   `-- scrapers/
|       |-- businessstandard_scraper.py
|       |-- economictimes_scraper.py
|       |-- financialexpress_scraper.py
|       |-- moneycontrol_scraper.py
|       `-- nifty_yfinance_scraper.py
|-- dataset/
|   |-- raw_dataset/
|   |-- stock_dataset/
|   |-- news_segregation_checkpoints/
|   |-- processed_news_dataset.csv
|   |-- tier_segregated_news.csv
|   |-- news_sentiment.csv
|   `-- tft_ready.csv
`--- artifacts/
    |-- tft/
    |-- tft_tune/
    `-- visualizations/
```

---

## Pipeline Flow (Stage by Stage)

### Stage 0: Data Collection

- News scraping from Moneycontrol, Financial Express, Economic Times, Business Standard.
- Ticker OHLCV scraping from Yahoo Finance for Nifty constituents.

### Stage 1: Preprocessing

- `src/0_preprocess_news.py`
- Merges `*_raw.csv`, cleans text, validates dates, deduplicates by URL, applies length filtering.
- Output: `dataset/processed_news_dataset.csv`.

### Stage 2: News Segregation

Two supported pipelines:

- `src/1a_openai_news_segregation.py` (OpenAI API)
- `src/1b_qwen_news_segregation.py` (local Ollama Qwen)

Both classify each article into `direct`, `sectoral`, `global`, with checkpoint/resume support.

Expected downstream output:
- `dataset/tier_segregated_news.csv`

### Stage 3: Sentiment Scoring

- `src/2_finbert_sentiment.py`
- FinBERT sentiment over all three news channels, with unique-text dedup and sliding windows for long texts.
- Output: `dataset/news_sentiment.csv`.

### Stage 4: Feature Engineering

- `src/3_feature_engineering.py`
- Merges sentiment + ticker data, adds technical features (RSI, MACD, volatility, MA ratios), builds target and `time_idx`.
- Output: `dataset/tft_ready.csv`.

### Stage 5: Modeling (TFT)

- `src/4_tft_hpt_train_test.py`
- Unified menu for tune/train/test/inference.
- Artifacts written under `artifacts/tft_tune` and `artifacts/tft`.

### Stage 6: Visualization

- `src/5_tft_visualize.py`
- Generates:
  - `best_window_actual_vs_predicted.png`
  - `per_window_metrics.png`
  - `best_window_stock_mape.png`

---

## Dataset Snapshot (Current Files)

Observed local row counts (excluding header):

- `dataset/raw_dataset/businessstandard_raw.csv`: 49,046
- `dataset/raw_dataset/economictimes_raw.csv`: 90,417
- `dataset/raw_dataset/financialexpress_raw.csv`: 156,214
- `dataset/raw_dataset/moneycontrol_raw.csv`: 15,855
- `dataset/processed_news_dataset.csv`: 309,042
- `dataset/tier_segregated_news.csv`: 125,510
- `dataset/news_sentiment.csv`: 125,510
- `dataset/stock_dataset/nifty50_ticker.csv`: 74,438
- `dataset/tft_ready.csv`: 71,938

Schema progression:

- Raw: `date,title,news,url`
- Segregated: `date,symbol,direct_news,sectoral_news,global_news`
- Sentiment: 9 sentiment probabilities + 3 count columns
- TFT-ready: market features + sentiment features + `target_pct_change` + `time_idx`

---

## Environment and Dependency Installation (Conflict-Safe)

All commands must run in conda env `nlp_project`.

```bash
conda activate nlp_project
```

### Why these pinned versions

TFT reliability here depends on version alignment among:

- `torch`
- `lightning` / `pytorch-lightning`
- `pytorch-forecasting`
- `optuna` + `optuna-integration`
- `numpy` / `pandas` / `scikit-learn`

`pytorch-forecasting==1.1.1` is sensitive to major-version drift, so this README keeps a locked compatibility set.

### Exact installation commands

```bash
# Optional clean rebuild
conda remove -n nlp_project --all -y

# Create env
conda create -n nlp_project python=3.10 pip setuptools wheel -y
conda activate nlp_project
conda config --env --set channel_priority strict

# 1) Core scientific + scraper dependencies (defaults priority)
conda install -y \
  numpy=1.26.4 pandas=2.2.2 scipy=1.15.3 scikit-learn=1.4.2 \
  matplotlib=3.8.* tqdm pyarrow=16.0.0 \
  requests aiohttp beautifulsoup4 lxml selenium python-dotenv yfinance

# 2) PyTorch CUDA stack
conda install -y -c pytorch -c nvidia \
  pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 pytorch-cuda=11.8

# 3) Forecasting/tuning/NLP stack (conda-forge fallback layer)
conda install -y -c conda-forge \
  lightning=2.2.4 pytorch-lightning=2.2.4 pytorch-forecasting=1.1.1 \
  optuna=3.6.1 optuna-integration=3.6.0 ta=0.11.0 \
  transformers openai aiofiles curl_cffi

# 4) Pip last-resort package used by the Qwen pipeline
pip install ollama

# 5) Dependency conflict check
python -m pip check

# 6) Quick compatibility check
python -c "import torch, lightning, pytorch_lightning, pytorch_forecasting, optuna, optuna_integration, numpy, pandas, sklearn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available()); print('lightning', lightning.__version__, 'pl', pytorch_lightning.__version__, 'pf', pytorch_forecasting.__version__)"
```

---

## How to Run (End-to-End Commands)

```bash
conda activate nlp_project
```

### 0) Optional: refresh source datasets

```bash
python src/scrapers/moneycontrol_scraper.py
python src/scrapers/financialexpress_scraper.py
python src/scrapers/economictimes_scraper.py
python src/scrapers/businessstandard_scraper.py
python src/scrapers/nifty_yfinance_scraper.py --index nifty50 --start 2020-01-01 --end 2026-03-31
```

### 1) Preprocess merged news

```bash
python src/0_preprocess_news.py
```

### 2A) Run OpenAI segregation

```bash
python src/1a_openai_news_segregation.py
```

### 2B) Run Qwen segregation [If OpenAI API key is not available]

Start Ollama separately:

```bash
ollama serve
ollama pull qwen2.5:3b
```

Then run:

```bash
python src/1b_qwen_news_segregation.py
```

Menu:
- `1` NER only
- `2` CSV only
- `3` Full pipeline

### 3) Run FinBERT sentiment

```bash
python src/2_finbert_sentiment.py
```

### 4) Build TFT-ready features

```bash
python src/3_feature_engineering.py
```

### 5) TFT run (menu-based)

```bash
python src/4_tft_hpt_train_test.py --data-path dataset/tft_ready.csv --windows 7,10,15,30
```

Menu:
- `0` tune + train + test
- `1` train + test
- `2` inference only
- `3` exit

### 6) Generate plots

```bash
python src/5_tft_visualize.py \
  --artifact-root artifacts/tft \
  --ticker-path dataset/stock_dataset/nifty50_ticker.csv \
  --windows 7,10,15,30 \
  --output-dir artifacts/visualizations \
  --dpi 300
```

---

## What Each Main File Does

### Core pipeline

- `src/0_preprocess_news.py`: merges and cleans raw article files.
- `src/1a_openai_news_segregation.py`: OpenAI-based article extraction/classification + checkpointed aggregation.
- `src/1b_qwen_news_segregation.py`: Qwen/Ollama-based article extraction/classification + checkpointed aggregation.
- `src/2_finbert_sentiment.py`: FinBERT sentiment feature generation.
- `src/3_feature_engineering.py`: feature engineering and final training table creation.
- `src/4_tft_hpt_train_test.py`: unified TFT menu runner for tune/train/test/inference.
- `src/5_tft_visualize.py`: final metric and chart generation.

### Scrapers

- `src/scrapers/moneycontrol_scraper.py`: async scraper with TLS impersonation.
- `src/scrapers/financialexpress_scraper.py`: async + process-pool scraper with checkpointing.
- `src/scrapers/economictimes_scraper.py`: sitemap-based historical scraper.
- `src/scrapers/businessstandard_scraper.py`: hybrid `curl_cffi` + Selenium scraper.
- `src/scrapers/nifty_yfinance_scraper.py`: index constituent + price downloader.

---

## Current Result Snapshot (from Existing Artifacts)

From `artifacts/tft/metrics_summary.csv`:

- W7 MAPE: `1.0487`
- W10 MAPE: `1.0582`
- W15 MAPE: `1.0470` (best in current snapshot)
- W30 MAPE: `1.0574`
- W15 Directional Accuracy: `0.5061`
- W15 F1 (`f1_up`): `0.5014`

Tuning summary is available in `artifacts/tft_tune/tuning_summary.csv`.


