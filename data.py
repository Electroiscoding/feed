import numpy as np, torch
from sentence_transformers import SentenceTransformer

rng = np.random.default_rng(0)
NC, NS, PER_DAY, DAYS, NU, K, H, POOL = 5, 6, 100, 60, 3000, 12, 20, 14
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
TPL = ["{a} and {b}: what {c} changes", "Why everyone is talking about {a}",
       "My honest take on {a}, {b} and {c}", "{a}, {b}, {c}: a quick breakdown",
       "Behind the scenes of {a} with {b}", "Top 5 things to know about {a} and {c}"]

# ---- items ----
NI = PER_DAY * DAYS
cat = rng.choice(NC, NI, p=[.25, .2, .15, .25, .15]); sub = rng.integers(0, NS, NI)
quality = rng.lognormal(0, .6, NI); g = cat * NS + sub

def post(c, s):
    w = list(rng.choice(KW[c][s], 5, replace=False))
    if rng.random() < .15: w[0] = rng.choice(KW[c][rng.integers(NS)])
    if rng.random() < .05: w[1] = rng.choice(KW[rng.integers(NC)][rng.integers(NS)])
    return rng.choice(TPL).format(a=w[0], b=w[1], c=w[2]) + f" Covers {w[3]} and {w[4]} in detail."

texts = [post(c, s) for c, s in zip(cat, sub)]
emb = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
F = np.hstack([np.eye(NC)[cat], emb]).astype(np.float32)          # 5 + 384 = 389

# ---- users & sessions ----
AGE = np.array([[1,0,1,2,1], [2,1,0,1,1], [0,2,1,0,1], [0,1,2,0,2]])
R = {k: [] for k in ["aff", "hist", "demo", "pos", "neg", "w", "day"]}
for u in range(NU):
    a = rng.choice(4, p=[.3, .35, .2, .15])
    cp = rng.dirichlet(.5 + .5 * AGE[a])
    taste = (cp[:, None] * rng.dirichlet(np.full(NS, .4), NC)).ravel()   # hidden
    match = taste / taste.max()
    n = int(np.clip(rng.lognormal(1.8, .9), 1, 40))
    days = np.sort(rng.integers(rng.integers(0, 50), DAYS, n))
    aff, hist, last = np.zeros((NC, 3)), [], days[0]
    for d in days:
        aff *= .93 ** (d - last); last = d
        pool = np.arange(max(0, (d - POOL) * PER_DAY), (d + 1) * PER_DAY)
        imp = rng.choice(pool, K, replace=False, p=quality[pool] / quality[pool].sum())
        m = match[g[imp]]
        eng = rng.random(K) < 1 / (1 + np.exp(2.5 - 4 * m - .4 * np.log(quality[imp])))
        if not eng.any(): continue
        me, ie = m[eng], imp[eng]
        watch = rng.beta(1 + 3 * me, 1.5)
        like = rng.random(len(ie)) < .1 + .5 * me
        com = rng.random(len(ie)) < .02 + .15 * me
        if not eng.all():                                   # rows use state BEFORE this session
            snap = np.log1p(aff).ravel(); h = hist[-H:]; hp = [-1] * (H - len(h)) + h
            for j, i in enumerate(ie):
                for k, v in zip(R, [snap, hp, np.eye(4)[a], i, rng.choice(imp[~eng]),
                                    1 + watch[j] + 2 * like[j] + 3 * com[j], d]): R[k].append(v)
        for j, i in enumerate(ie):
            aff[cat[i]] += [like[j], watch[j], com[j]]; hist.append(i)

T = lambda k, t: torch.tensor(np.array(R[k]), dtype=t)
torch.save(dict(F=torch.tensor(F), aff=T("aff", torch.float32), hist=T("hist", torch.long),
                demo=T("demo", torch.float32), pos=T("pos", torch.long), neg=T("neg", torch.long),
                w=T("w", torch.float32), day=T("day", torch.long)), "data.pt")
print(f"{NI} posts, {NU} users, {len(R['w'])} training rows, "
      f"{np.mean([(np.array(h) < 0).all() for h in R['hist']]):.0%} rows from users with empty history")
