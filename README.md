# Feed Recommendation System

A lightweight, production-style two-tower recommendation feed system with synthetic simulation, cold-start handling, and real text embeddings.

## Project Structure

```
feed/
├── requirements.txt   # torch, numpy, sentence-transformers
├── data.py            # synthetic world → real embeddings → data.pt
├── model.py           # two towers + loss
├── train.py           # train, evaluate, export user/item towers
└── serve.py           # inference, freshness, diversity & exploration
```

## Setup & Running

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate data and embeddings:**
   ```bash
   python data.py
   ```

3. **Train the two-tower models:**
   ```bash
   python train.py
   ```

4. **Run serving and recommendations:**
   ```bash
   python serve.py
   ```
