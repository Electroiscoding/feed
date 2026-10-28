#!/usr/bin/env python3
"""
Feed Demo - You are the lab rat. Like or Skip posts and watch the model adapt.
Run: python3 demo.py
"""
import numpy as np, torch
import tkinter as tk
from tkinter import ttk

# ── Rebuild post texts (same seed as data.py so texts are identical) ──────────
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

# ── Load trained towers ───────────────────────────────────────────────────────
from model import ItemTower, UserTower
D = torch.load("data.pt"); F = D["F"]
_it = ItemTower(); _ut = UserTower()
_it.load_state_dict(torch.load("item_tower.pt"))
_ut.load_state_dict(torch.load("user_tower.pt"))
_it.eval(); _ut.eval()
with torch.no_grad(): IV = _it(F)
CAT_T = F[:,:5].argmax(1); DAY_T = torch.arange(len(F)) // PER_DAY

@torch.no_grad()
def recommend(aff15, hist_ids, demo4, seen, today, n=10, k=200, fresh=0.01, div=0.05, eps=0.1):
    h = F[hist_ids].mean(0) if hist_ids else torch.zeros(F.shape[1])
    u = _ut(aff15[None], h[None], demo4[None])[0]
    ok = (DAY_T <= today) & (DAY_T >= today - POOL)
    if seen: ok[list(seen)] = False
    sc = (IV @ u - fresh*(today - DAY_T)).masked_fill(~ok, -1e9)
    cand = sc.topk(min(k, int(ok.sum()))).indices.tolist()
    out, cnt = [], torch.zeros(5)
    for _ in range(n):
        if torch.rand(1) < eps:
            pool = ok.clone()
            for o in out: pool[o] = False
            v = pool.nonzero().squeeze(1)
            if not len(v): break
            out.append(v[torch.randint(len(v),(1,))].item()); cnt[CAT_T[out[-1]]] += 1; continue
        rem = [c for c in cand if c not in out]
        if not rem: break
        best = max(rem, key=lambda c: sc[c].item() - div*cnt[CAT_T[c]].item())
        out.append(best); cnt[CAT_T[best]] += 1
    return out

# ── Theme ─────────────────────────────────────────────────────────────────────
TODAY       = 59
CAT_NAMES   = ["🖥️ Tech","💼 Business","🎨 Arts","🎬 Entertainment","⚽ Sports"]
CAT_COLORS  = ["#4A90D9","#2ECC71","#9B59B6","#E74C3C","#F39C12"]
CAT_LIGHT   = ["#EBF5FB","#EAFAF1","#F5EEF8","#FDEDEC","#FEF9E7"]

# ── App ───────────────────────────────────────────────────────────────────────
class FeedApp:
    def __init__(self, root):
        self.root = root
        root.title("🧠 Feed Demo — You are the Lab Rat 🐀")
        root.geometry("980x700"); root.configure(bg="#0f0f23"); root.resizable(True,True)

        self.aff_raw = np.zeros((NC,3), dtype=np.float32)
        self.demo    = torch.tensor(np.eye(4)[1], dtype=torch.float32)
        self.hist    = []; self.seen = set(); self.queue = []
        self.n_likes = 0; self.n_skips = 0
        self.cat_likes = [0]*NC; self.current_id = None

        self._build(); self._refill(); self._next()

    def _build(self):
        r = self.root
        # header
        hdr = tk.Frame(r, bg="#16213e", pady=8); hdr.pack(fill="x")
        tk.Label(hdr, text="🧠  Feed Recommendation Demo",
                 font=("Helvetica",17,"bold"), bg="#16213e", fg="white").pack(side="left",padx=18)
        self.status = tk.Label(hdr, text="🧊 Cold start — no history yet",
                               font=("Helvetica",10), bg="#16213e", fg="#8888bb")
        self.status.pack(side="right", padx=18)

        body = tk.Frame(r, bg="#0f0f23"); body.pack(fill="both", expand=True, padx=10, pady=8)

        # ─ left (post card) ─
        L = tk.Frame(body, bg="#0f0f23"); L.pack(side="left", fill="both", expand=True, padx=(0,8))

        self.card = tk.Frame(L, bg="white", highlightthickness=3, highlightbackground="#4A90D9")
        self.card.pack(fill="both", expand=True)

        self.badge = tk.Label(self.card, text="", font=("Helvetica",10,"bold"),
                               fg="white", padx=14, pady=6, anchor="w")
        self.badge.pack(fill="x")

        self.post_txt = tk.Text(self.card, font=("Helvetica",14), wrap="word", bd=0,
                                 padx=20, pady=20, state="disabled", height=7,
                                 relief="flat", highlightthickness=0)
        self.post_txt.pack(fill="both", expand=True)

        self.meta = tk.Label(self.card, text="", font=("Helvetica",9),
                              fg="#999", anchor="w", padx=20, pady=6)
        self.meta.pack(fill="x")

        bf = tk.Frame(L, bg="#0f0f23"); bf.pack(fill="x", pady=(5,0))
        bkw = dict(font=("Helvetica",14,"bold"), bd=0, relief="flat", pady=14, cursor="hand2")
        tk.Button(bf, text="👍  Like",  bg="#1db954", fg="white",
                  command=self.like, **bkw).pack(side="left", fill="x", expand=True, padx=(0,5))
        tk.Button(bf, text="⏭  Skip",  bg="#e74c3c", fg="white",
                  command=self.skip, **bkw).pack(side="left", fill="x", expand=True, padx=(5,0))

        pf = tk.Frame(L, bg="#0f0f23", pady=5); pf.pack(fill="x")
        tk.Label(pf, text="Interactions:", bg="#0f0f23", fg="#555577",
                 font=("Helvetica",9)).pack(side="left")
        self.prog = ttk.Progressbar(pf, maximum=50, length=200)
        self.prog.pack(side="left", padx=6)
        self.prog_lbl = tk.Label(pf, text="0 / 50", bg="#0f0f23", fg="#555577",
                                  font=("Helvetica",9)); self.prog_lbl.pack(side="left")

        # ─ right (stats panel) ─
        R = tk.Frame(body, bg="#16213e", width=240,
                     highlightthickness=1, highlightbackground="#2d2d50")
        R.pack(side="right", fill="y"); R.pack_propagate(False)

        tk.Label(R, text="Your Taste Profile", font=("Helvetica",11,"bold"),
                 bg="#16213e", fg="white", pady=10).pack(fill="x", padx=12)

        self.bars = []; self.pcts = []
        for i in range(NC):
            f = tk.Frame(R, bg="#16213e", pady=4); f.pack(fill="x", padx=10)
            tk.Label(f, text=CAT_NAMES[i], font=("Helvetica",8,"bold"), bg="#16213e",
                     fg=CAT_COLORS[i], width=17, anchor="w").pack(side="left")
            bg = tk.Frame(f, bg="#252545", height=14, width=92); bg.pack(side="left", padx=3)
            bg.pack_propagate(False)
            bar = tk.Frame(bg, bg=CAT_COLORS[i], height=14, width=1)
            bar.place(x=0, y=0, relheight=1.0)
            pct = tk.Label(f, text="0%", font=("Helvetica",8), bg="#16213e",
                            fg="#888899", width=4); pct.pack(side="left")
            self.bars.append(bar); self.pcts.append(pct)

        tk.Frame(R, bg="#2d2d50", height=1).pack(fill="x", padx=10, pady=7)
        tk.Label(R, text="Recent activity", font=("Helvetica",9,"bold"),
                 bg="#16213e", fg="white").pack(fill="x", padx=12)

        self.log = tk.Text(R, font=("Helvetica",8), bg="#16213e", fg="#aaaacc",
                            bd=0, state="disabled", wrap="word", padx=8, pady=3,
                            height=11, relief="flat", highlightthickness=0)
        self.log.pack(fill="both", expand=True, padx=4)

        tk.Frame(R, bg="#2d2d50", height=1).pack(fill="x", padx=10, pady=4)
        self.stat_lbl = tk.Label(R, text="", font=("Helvetica",8), bg="#16213e",
                                  fg="#8888aa", justify="left", padx=12, pady=4, anchor="w")
        self.stat_lbl.pack(fill="x")

    # ── logic ──────────────────────────────────────────────────────────────────
    def _aff_t(self):
        return torch.tensor(np.log1p(self.aff_raw).ravel(), dtype=torch.float32)

    def _refill(self):
        recs = recommend(self._aff_t(), self.hist, self.demo, self.seen, TODAY, n=12)
        for r in recs:
            if r not in self.queue: self.queue.append(r)

    def _next(self):
        if len(self.queue) < 4: self._refill()
        if not self.queue: return
        self.current_id = self.queue.pop(0)
        self.seen.add(self.current_id)
        c = int(_cat[self.current_id]); d = self.current_id // PER_DAY
        self.badge.config(text=f"  {CAT_NAMES[c]}", bg=CAT_COLORS[c])
        self.card.config(highlightbackground=CAT_COLORS[c])
        self.post_txt.config(state="normal", bg=CAT_LIGHT[c])
        self.post_txt.delete("1.0","end")
        self.post_txt.insert("1.0", TEXTS[self.current_id])
        self.post_txt.config(state="disabled")
        self.meta.config(text=f"📅 Day {d}  •  Post #{self.current_id}  •  ⭐ quality {_qual[self.current_id]:.2f}",
                         bg=CAT_LIGHT[c])

    def _interact(self, liked):
        c = int(_cat[self.current_id])
        if liked:
            self.aff_raw[c,0] += 1; self.aff_raw[c,1] += 0.7
            self.hist = (self.hist + [self.current_id])[-H:]
            self.cat_likes[c] += 1; self.n_likes += 1; icon = "👍"
        else:
            self.n_skips += 1; icon = "⏭"

        # activity log
        self.log.config(state="normal")
        self.log.insert("1.0", f"{icon} {CAT_NAMES[c][:10]}  {TEXTS[self.current_id][:36]}…\n")
        self.log.config(state="disabled")

        # taste bars
        total = sum(self.cat_likes) or 1
        for i in range(NC):
            p = self.cat_likes[i] / total
            self.bars[i].place(width=max(1, int(p*92)))
            self.pcts[i].config(text=f"{int(p*100)}%")

        # warm / cold label
        is_warm = len(self.hist) >= 3
        self.status.config(
            text=f"{'🔥 Warm' if is_warm else '🧊 Cold'}  •  {len(self.hist)} engagements  •  {len(self.seen)} seen",
            fg="#1db954" if is_warm else "#8888bb")

        n_tot = self.n_likes + self.n_skips
        top = self.cat_likes.index(max(self.cat_likes)) if any(self.cat_likes) else -1
        self.stat_lbl.config(text=(
            f"Liked:   {self.n_likes}\n"
            f"Skipped: {self.n_skips}\n"
            f"Like rate: {self.n_likes/max(1,n_tot)*100:.0f}%\n"
            f"Top interest: {CAT_NAMES[top] if top>=0 else '—'}"
        ))
        self.prog["value"] = min(n_tot, 50)
        self.prog_lbl.config(text=f"{n_tot} / 50")
        self._next()

    def like(self): self._interact(True)
    def skip(self): self._interact(False)

if __name__ == "__main__":
    root = tk.Tk()
    FeedApp(root)
    root.mainloop()
