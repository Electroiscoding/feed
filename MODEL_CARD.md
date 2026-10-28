---
license: apache-2.0
tags:
- recommender-system
- two-tower
- deep-learning
- retrieval
- candidate-generation
- cold-start
- feed-ranking
library_name: pytorch
pipeline_tag: feature-extraction
---

# Two-Tower Recommendation Model (ItemTower & UserTower)

A production-style, dual-tower deep learning recommendation system designed for real-time candidate generation, feed ranking, and cold-start personalization.

- **Architecture:** Decoupled Two-Tower MLP with cosine metric learning
- **Item Backbone:** 5-category one-hot + 384-dim `all-MiniLM-L6-v2` dense text embeddings
- **User Inputs:** Logged affinity matrix (`log(1 + x)` decayed interactions), history centroid, demographic age buckets
- **Total Parameters:** **270,336** (ItemTower: 132,736 · UserTower: 137,600)
- **Inference Latency:** **< 1.0 ms** on standard CPU

---

## ⚡ Hyper-to-the-Point: Inputs & Outputs

### 1. ItemTower (`item_tower.pt`)
Embeds posts/items into a shared 128-dimensional hypersphere.

| Direction | Parameter | Tensor Shape | Dtype | Range & Description |
|---|---|---|---|---|
| **INPUT** | `x` | `[Batch, 389]` | `torch.float32` | Concatenation of Category + Text embedding: |
| | ↳ `x[:, 0:5]` | `[Batch, 5]` | `torch.float32` | One-hot category: `[Tech, Business, Arts, Entertainment, Sports]` |
| | ↳ `x[:, 5:389]` | `[Batch, 384]` | `torch.float32` | Dense sentence embedding from `sentence-transformers/all-MiniLM-L6-v2` (unit length) |
| **OUTPUT** | `item_emb` | `[Batch, 128]` | `torch.float32` | Unit L2-normalized vector on hypersphere (‖v‖₂ = 1.0) |

---

### 2. UserTower (`user_tower.pt`)
Embeds live user context and preferences into the same 128-dimensional hypersphere.

| Direction | Parameter | Tensor Shape | Dtype | Range & Description |
|---|---|---|---|---|
| **INPUT 1** | `aff` | `[Batch, 15]` | `torch.float32` | `log(1 + x)` transformed interactions across 5 categories × 3 action signals: `[likes, watch_time, comments]` |
| **INPUT 2** | `hist` | `[Batch, 389]` | `torch.float32` | Mean 389-dim feature vector of the user's last N ≤ 20 engaged posts. *(For **cold users** with 0 engagements, pass all-zeros: `torch.zeros(389)`)* |
| **INPUT 3** | `demo` | `[Batch, 4]` | `torch.float32` | One-hot encoded age bracket (`[1,0,0,0]`, `[0,1,0,0]`, etc.) |
| **OUTPUT** | `user_emb` | `[Batch, 128]` | `torch.float32` | Unit L2-normalized vector on hypersphere (‖u‖₂ = 1.0) |

---

## 📐 Relevance & Serving Formulas

### 1. Cosine Retrieval with Freshness Penalty

$$
\text{score}_i = \mathbf{u} \cdot \mathbf{v}_i - \lambda_{\text{fresh}} \cdot (\text{today} - \text{day}_i)
$$

- `u`: 128-dim User Tower vector
- `v_i`: 128-dim Item Tower vector for candidate post `i`
- `λ_fresh = 0.01`: small linear penalty per day of post age (applied at serve time)

---

### 2. Greedy Diversity Re-ranking

$$
\text{pick}^* = \arg\max_{c \notin \text{picked}} \left( \text{score}_c - \lambda_{\text{div}} \cdot \text{count}[\text{category}_c] \right)
$$

- `λ_div = 0.05`: penalizes a category each time it has already been picked in the current recommendation session, preventing the feed from becoming a single-topic echo chamber.

---

### 3. Epsilon-Greedy Exploration (ε = 0.1)

- With probability **10% (ε = 0.1)**, an item is sampled uniformly from the active 14-day pool.
- This discovers emerging interests and logs unbiased exploration data for future model retraining.

---

## 🚀 Google Colab Ready Code Snippet

Copy and paste this into any fresh [Google Colab](https://colab.research.google.com/) notebook:

```python
# Step 1: Install dependencies
!pip install -q torch sentence-transformers huggingface_hub

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

REPO_ID = "Phase-Technologies/two-tower-feed-recsys"

# Step 2: Define Model Architectures
mlp = lambda i, h, o: nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))

class ItemTower(nn.Module):
    def __init__(self, item_in=389, out=128):
        super().__init__()
        self.net = mlp(item_in, 256, out)
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class UserTower(nn.Module):
    def __init__(self, item_in=389, aff=15, demo=4, out=128):
        super().__init__()
        self.net = mlp(item_in + aff + demo, 256, out)
    def forward(self, aff, hist, demo):
        return F.normalize(self.net(torch.cat([aff, hist, demo], -1)), dim=-1)

# Step 3: Download Weights from Hugging Face
item_weights = hf_hub_download(repo_id=REPO_ID, filename="item_tower.pt")
user_weights = hf_hub_download(repo_id=REPO_ID, filename="user_tower.pt")

item_tower = ItemTower()
user_tower = UserTower()
item_tower.load_state_dict(torch.load(item_weights, map_location="cpu", weights_only=True))
user_tower.load_state_dict(torch.load(user_weights, map_location="cpu", weights_only=True))
item_tower.eval(); user_tower.eval()
print("✅ Two-Tower models downloaded and loaded!")

# Step 4: Embed Candidate Posts using all-MiniLM-L6-v2
embedder = SentenceTransformer("all-MiniLM-L6-v2")

posts_data = [
    {"cat": 0, "text": "Next-gen GPUs, wafer yields, and AI hardware acceleration"},
    {"cat": 0, "text": "Python compiler optimizations and modern rust debugging"},
    {"cat": 1, "text": "Venture capital, startup pitch decks, and angel seed funding"},
    {"cat": 1, "text": "Real estate mortgage rates and housing market rental yields"},
    {"cat": 2, "text": "Classical music, vinyl records, and gallery art exhibitions"},
    {"cat": 3, "text": "Box office record breakers, blockbuster sequels, and film trailers"},
    {"cat": 4, "text": "Premier league derby goals, tactical managers, and transfer window"}
]

cat_one_hots = F.one_hot(torch.tensor([p["cat"] for p in posts_data]), num_classes=5).float()
text_embs = torch.tensor(embedder.encode([p["text"] for p in posts_data], normalize_embeddings=True))
item_features = torch.cat([cat_one_hots, text_embs], dim=-1) # [7, 389]

with torch.no_grad():
    IV = item_tower(item_features) # [7, 128]

# Step 5: Simulate a User with strong Tech preference (Category 0)
aff = torch.zeros(15)
aff[0] = 3.5  # Heavy Tech likes
aff[1] = 2.0  # High Tech watch time
hist = item_features[0].unsqueeze(0) # engaged with GPU post
demo = torch.tensor([[0.0, 1.0, 0.0, 0.0]]) # Age bucket 1

with torch.no_grad():
    user_vec = user_tower(aff.unsqueeze(0), hist, demo)[0] # [128]

# Step 6: Rank candidates via Cosine Similarity
scores = (IV @ user_vec).tolist()
ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

print("\n--- Personalized Feed Recommendations ---")
for rank, idx in enumerate(ranked_indices, 1):
    cat_name = ["Tech", "Business", "Arts", "Entertainment", "Sports"][posts_data[idx]["cat"]]
    print(f"#{rank} [{cat_name}] (Score: {scores[idx]:.4f}) -> {posts_data[idx]['text']}")
```

---

## 📊 Evaluation & Accuracy Benchmarks

Evaluated over held-out future days (`days 52–59`) against live pools of **~1,500 active candidates**:

| Cohort | Samples (N) | Recall@10 | Recall@50 | Recall@100 | MRR | Lift vs Random |
|---|---|---|---|---|---|---|
| **ALL Users** | 5,000 | **0.015** | **0.082** | **0.147** | **0.0098** | **2.2×** |
| **COLD Users** | 173 | **0.017** | **0.052** | **0.092** | **0.0090** | **2.5×** (@10) |
| **WARM Users** | 4,827 | **0.015** | **0.083** | **0.149** | **0.0099** | **2.2×** |
| *Random Baseline* | — | *0.007* | *0.033* | *0.067* | *0.0044* | *1.0×* |

- **Recall@100 = 14.7%** (Beats random 0.067 by **2.2×**).
- **Cold-Start Lift @10 = 2.5×**: When user history is zero, the model leverages demographic priors and semantic text representations to reliably surface relevant candidates.

---

## 🛠️ Model Architecture & Training Details

### Contrastive InfoNCE Loss

$$
\mathcal{L} = -\sum_{i} w_i \log \frac{\exp(\mathbf{u}_i \cdot \mathbf{v}_i^+ / \tau)}{\sum_{j \in \mathcal{B} \cup \mathcal{N}_{\text{logged}}} \exp(\mathbf{u}_i \cdot \mathbf{v}_j / \tau)}
$$

- **Learned Temperature:** $\tau$ with initial $\log \alpha = 2.64$
- **Hard Negative Mining:** In addition to standard in-batch negatives ($\mathcal{B}$), actual un-clicked/skipped impressions from the same session ($\mathcal{N}_{\text{logged}}$) are treated as explicit negative items.
- **Optimizer:** `AdamW(lr=2e-3, weight_decay=1e-4)`.
- **Training Epochs:** 15 epochs across 51,087 interaction rows.
