# Semantic Plagiarism Detection Agent

A lightweight, robust Natural Language Processing (NLP) system designed to detect semantic plagiarism, structural sentence rewrites, and synonym swapping where traditional exact-match checkers fail.

## Features
- **Sentence Tokenization**: Regex-based sentence boundary detection and normalization.
- **Hybrid Semantic Vectorizer**: Combines lexical word frequencies with sub-word character 3-grams to capture morphological root similarity and synonym shifts.
- **Cosine Distance Metric**: Computes standard pairwise cosine similarity across latent vector counters.
- **Dynamic Sensitivity Slider**: Allows evaluators to calibrate the threshold between strict paraphrase matching and broad thematic alignment.
- **Audit Reporting**: Real-time KPI analytics and downloadable `.txt` audit reports.

## Tech Stack
- **Language**: Python 3.10+
- **Framework**: Streamlit
- **Math/NLP**: Pure Python `math`, `collections.Counter`, and regex `re`

## Running Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/connectrehanshaik/Codethon-thepioeers.git](https://github.com/connectrehanshaik/Codethon-thepioeers.git)
   cd Codethon-thepioeers
