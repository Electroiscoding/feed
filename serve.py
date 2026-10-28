import torch
from model import ItemTower, UserTower

D = torch.load("data.pt"); F = D["F"]; PER_DAY, POOL = 100, 14
it, ut = ItemTower(), UserTower()
it.load_state_dict(torch.load("item_tower.pt")); ut.load_state_dict(torch.load("user_tower.pt"))
it.eval(); ut.eval()
with torch.no_grad(): IV = it(F)                      # item tower runs once, offline
CAT = F[:, :5].argmax(1); DAY = torch.arange(len(F)) // PER_DAY

@torch.no_grad()
def recommend(aff, hist_ids, demo, seen, today, n=20, k=200, fresh=0.01, div=0.05, eps=0.1):
    """aff (15, log1p'd), hist_ids: last-N engaged post ids, demo (4), seen: ids to exclude."""
    h = F[hist_ids].mean(0) if len(hist_ids) else torch.zeros(F.shape[1])
    u = ut(aff[None], h[None], demo[None])[0]                          # the only live model call
    ok = (DAY <= today) & (DAY >= today - POOL)                         # live posts
    ok[list(seen)] = False
    s = (IV @ u - fresh * (today - DAY)).masked_fill(~ok, -1e9)         # retrieval + freshness
    cand = s.topk(k).indices.tolist()
    out, cnt = [], torch.zeros(5)
    for _ in range(n):                                                  # diversity: greedy, penalise repeats
        if torch.rand(1) < eps:                                         # exploration slot
            pool = ok.clone(); pool[out] = False
            out.append(pool.nonzero()[torch.randint(int(pool.sum()), (1,))].item()); cnt[CAT[out[-1]]] += 1
            continue
        best = max((c for c in cand if c not in out), key=lambda c: s[c] - div * cnt[CAT[c]])
        out.append(best); cnt[CAT[best]] += 1
    return out

if __name__ == "__main__":
    for name, ix in [("cold", (D["hist"] < 0).all(1).nonzero()[-1].item()),
                     ("warm", (D["hist"] >= 0).sum(1).argmax().item())]:
        h = D["hist"][ix]; h = h[h >= 0].tolist(); today = D["day"][ix].item()
        r = recommend(D["aff"][ix], h, D["demo"][ix], set(h), today)
        print(f"{name} user | engaged cats {CAT[h].tolist()} | recs by cat {CAT[r].tolist()}")
