# Multi-Horizon Stock Price Trajectory Forecasting

**Status:** 🚧 Work in Progress (Active Development)

## 📌 Overview
This project implements a hybrid quantitative-qualitative framework to forecast stock price trends over a future time horizon (e.g., next 7 days) rather than a single point prediction. It integrates **Quantitative Data** (OHLC prices) with **Qualitative Data** (Financial News) using a novel **Multi-Tiered Sentiment Analysis** approach.

## 🚀 Key Features
* **Hierarchical Sentiment Engine:** Segregates news into 4 tiers:
    1.  **Direct:** Company-specific news.
    2.  **Sectoral:** Industry-wide trends.
    3.  **Supply Chain:** Upstream supplier risks (Early Warning System).
    4.  **Global:** Macro-economic events.
* **Hybrid Architecture:** Combines **FinBERT** (for domain-specific sentiment embedding) and **LSTM/Seq2Seq** (for time-series forecasting).
* **Multi-Horizon Output:** Predicts a probabilistic price trajectory (trend) instead of a static value.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch, LSTM, Transformers (Hugging Face)
* **NLP:** FinBERT, Spacy (NER)
* **Data:** Yahoo Finance, Economic Times Scraper

## 📅 Roadmap
- [ ] Phase 1: Data Ingestion Pipeline (Scrapers & DB)
- [ ] Phase 2: NLP Feature Engineering (4-Tier Sentiment Logic)
- [ ] Phase 3: Model Training (Seq2Seq Architecture)
- [ ] Phase 4: Dashboard & Evaluation
