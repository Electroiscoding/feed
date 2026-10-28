"""
Feed Recommendation API
POST /recommend  → get ranked posts with metadata for a user
GET  /post/{id}  → get post text + category
GET  /health     → liveness and model check
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np, torch
from typing import Optional

# ── rebuild post texts (same seed as data.py) ────────────────────────────────
_rng = np.random.default_rng(0)
NC, NS, PER_DAY, DAYS, H, POOL = 5, 6, 100, 60, 20, 14
KW = [[s.split(",") for s in c] for c in [
 ["graphics cards,chip fabrication,benchmarks,transistors,wafer yields,overclocking",
  "python,rust,compilers,debugging,open source,APIs",
  "AI models,neural networks,chatbots,training data,fine-tuning,inference costs",
  "smartphones,battery life,camera sensors,foldables,iOS,Android updates",
  "cybersecurity,data breaches,passwords,encryption,phishing,zero-days",
  "robots,drones,3D printing,self-driving cars,electric scooters,smart homes"],
 ["startups,seed funding,venture capital,pitch decks,founders,unicorns",
  "stock market,earnings,interest rates,inflation,index funds,bonds",
  "marketing,branding,SEO,growth hacking,ad campaigns,influencers",
  "real estate,mortgages,rental yields,housing market,property flipping,landlords",
  "remote work,productivity,leadership,hiring,burnout,management",
  "crypto,bitcoin,blockchain,wallets,DeFi,token launches"],
 ["oil painting,watercolor,gallery openings,sculpture,art auctions,museums",
  "poetry,novels,book clubs,literary prizes,writing prompts,bestsellers",
  "classical music,jazz,opera,orchestras,vinyl records,live concerts",
  "street photography,film cameras,portraits,darkroom,photo essays,lenses",
  "architecture,interior design,ceramics,textiles,calligraphy,craft fairs",
  "theatre,ballet,contemporary dance,stage design,playwrights,improv"],
 ["blockbuster movies,box office,film trailers,sequels,directors,oscars",
  "streaming series,season finales,binge watching,plot twists,showrunners,spoilers",
  "pop music,album drops,world tours,music videos,charts,collaborations",
  "video games,speedruns,esports,indie games,console launches,game mods",
  "stand-up comedy,memes,sketches,viral videos,pranks,podcasts",
  "celebrity news,red carpet,fan theories,anime,K-pop,reality shows"],
 ["football,transfer window,premier league,goals,managers,derbies",
  "basketball,NBA playoffs,three-pointers,draft picks,dunks,coaching",
  "cricket,IPL,test matches,bowling,batting records,world cup",
  "running,marathons,strength training,calisthenics,cycling,recovery",
  "tennis,grand slams,serve technique,rankings,clay courts,rivalries",
  "formula one,pit stops,qualifying,motorbikes,rally,lap records"]]]
TPL = ["{a} and {b}: what {c} changes","Why everyone is talking about {a}",
       "My honest take on {a}, {b} and {c}","{a}, {b}, {c}: a quick breakdown",
       "Behind the scenes of {a} with {b}","Top 5 things to know about {a} and {c}"]
CAT_NAMES = ["Tech","Business","Arts","Entertainment","Sports"]

NI = PER_DAY * DAYS
_cat  = _rng.choice(NC, NI, p=[.25,.2,.15,.25,.15])
_sub  = _rng.integers(0, NS, NI)
_qual = _rng.lognormal(0, .6, NI)

def _post(c, s):
    w = list(_rng.choice(KW[c][s], 5, replace=False))
    if _rng.random() < .15: w[0] = _rng.choice(KW[c][_rng.integers(NS)])
    if _rng.random() < .05: w[1] = _rng.choice(KW[_rng.integers(NC)][_rng.integers(NS)])
    return _rng.choice(TPL).format(a=w[0],b=w[1],c=w[2]) + f" Covers {w[3]} and {w[4]} in detail."

TEXTS = [_post(c, s) for c, s in zip(_cat, _sub)]

# ── Load model weights & cache item embeddings ──────────────────────────────
from model import ItemTower, UserTower
D = torch.load("data.pt", weights_only=False); F = D["F"]
it = ItemTower(); ut = UserTower()
it.load_state_dict(torch.load("item_tower.pt", weights_only=True))
ut.load_state_dict(torch.load("user_tower.pt", weights_only=True))
it.eval(); ut.eval()

with torch.no_grad():
    IV = it(F)

CAT_T = F[:, :5].argmax(1)
DAY_T = torch.arange(len(F)) // PER_DAY
print(f"✅ Models loaded. {len(F)} posts ready for serving.")

app = FastAPI(title="Feed Recommendation API", version="1.0")

# ── schemas ───────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    aff: list[float]           # 15 floats: log1p(likes/watch/comment) per category
    hist_ids: list[int]        # last-N engaged post ids (empty for cold users)
    demo: list[float]          # 4 floats: one-hot age bucket [0-3]
    seen: list[int] = []       # post ids already shown (to exclude)
    today: int = 59            # current day index (0-59)
    n: Optional[int] = 20      # how many recs to return
    fresh: Optional[float] = 0.01
    div: Optional[float] = 0.05
    eps: Optional[float] = 0.1

class Post(BaseModel):
    id: int
    text: str
    category: str
    category_id: int
    day: int
    quality: float

class RecommendResponse(BaseModel):
    user_status: str
    total_recs: int
    posts: list[Post]

# ── endpoints ─────────────────────────────────────────────────────────────────
@app.post("/recommend", response_model=RecommendResponse)
@torch.no_grad()
def recommend(req: RecommendRequest):
    if len(req.aff) != 15:
        raise HTTPException(400, "aff must be length 15 (5 categories × 3 signals)")
    if len(req.demo) != 4:
        raise HTTPException(400, "demo must be length 4 (one-hot age bucket)")

    aff15 = torch.tensor(req.aff, dtype=torch.float32)
    demo4 = torch.tensor(req.demo, dtype=torch.float32)
    h = F[req.hist_ids].mean(0) if req.hist_ids else torch.zeros(F.shape[1])
    u = ut(aff15[None], h[None], demo4[None])[0]

    ok = (DAY_T <= req.today) & (DAY_T >= req.today - POOL)
    if req.seen:
        ok[list(req.seen)] = False

    sc = (IV @ u - req.fresh * (req.today - DAY_T)).masked_fill(~ok, -1e9)
    k = min(200, int(ok.sum()))
    cand = sc.topk(k).indices.tolist()

    out, cnt = [], torch.zeros(5)
    for _ in range(req.n):
        if torch.rand(1) < req.eps:
            pool = ok.clone()
            for o in out:
                pool[o] = False
            v = pool.nonzero().squeeze(1)
            if not len(v):
                break
            out.append(v[torch.randint(len(v), (1,))].item())
            cnt[CAT_T[out[-1]]] += 1
            continue
        rem = [c for c in cand if c not in out]
        if not rem:
            break
        best = max(rem, key=lambda c: sc[c].item() - req.div * cnt[CAT_T[c]].item())
        out.append(best)
        cnt[CAT_T[best]] += 1

    posts = [
        Post(
            id=i,
            text=TEXTS[i],
            category=CAT_NAMES[int(CAT_T[i])],
            category_id=int(CAT_T[i]),
            day=int(DAY_T[i]),
            quality=round(float(_qual[i]), 3)
        )
        for i in out
    ]

    status = "warm" if len(req.hist_ids) >= 3 else "cold"
    return RecommendResponse(user_status=status, total_recs=len(posts), posts=posts)

@app.get("/post/{post_id}", response_model=Post)
def get_post(post_id: int):
    if post_id < 0 or post_id >= NI:
        raise HTTPException(404, f"post_id must be between 0 and {NI-1}")
    return Post(
        id=post_id,
        text=TEXTS[post_id],
        category=CAT_NAMES[int(CAT_T[post_id])],
        category_id=int(CAT_T[post_id]),
        day=int(DAY_T[post_id]),
        quality=round(float(_qual[post_id]), 3)
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "posts_indexed": NI,
        "device": str(F.device),
        "model": "two-tower-recsys v1"
    }
