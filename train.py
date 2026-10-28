import torch
from model import TwoTower

torch.manual_seed(0)
D = torch.load("data.pt"); F = D["F"]; PER_DAY, POOL = 100, 14

def batch(ix):
    h = D["hist"][ix]; mk = (h >= 0).float().unsqueeze(-1)
    hist = (F[h.clamp(min=0)] * mk).sum(1) / mk.sum(1).clamp(min=1)    # mean of last-N engaged items
    return D["aff"][ix], hist, D["demo"][ix], D["pos"][ix], D["neg"][ix], D["w"][ix], F

tr = (D["day"] < 52).nonzero().squeeze(1); va = (D["day"] >= 52).nonzero().squeeze(1)
m = TwoTower(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)

@torch.no_grad()
def recall(k=100):
    """Where does the engaged post rank among all posts in that day's 14-day pool?"""
    m.eval(); IV = m.item(F); iday = torch.arange(len(F)) // PER_DAY
    ix = va[torch.randperm(len(va))[:5000]]
    aff, hist, demo, pos, *_ = batch(ix); day = D["day"][ix][:, None]
    sc = m.user(aff, hist, demo) @ IV.T
    sc = sc.masked_fill(~((iday[None] <= day) & (iday[None] >= day - POOL)), -1e9)
    rank = (sc > sc.gather(1, pos[:, None])).sum(1)
    cold = (D["hist"][ix] < 0).all(1); hit = (rank < k).float()
    base = k / ((iday[None] <= day) & (iday[None] >= day - POOL)).sum(1).float().mean()
    return hit.mean().item(), hit[cold].mean().item(), hit[~cold].mean().item(), base.item()

best = 0
for ep in range(15):
    m.train(); p = tr[torch.randperm(len(tr))]; tot = 0
    for i in range(0, len(p) - 511, 512):
        opt.zero_grad(); l = m.loss(*batch(p[i:i + 512])); l.backward(); opt.step(); tot += l.item()
    r, rc, rw, base = recall()
    print(f"ep {ep:2d} loss {tot / (len(p) // 512):.3f} | recall@100 all {r:.3f} cold {rc:.3f} warm {rw:.3f} (random {base:.3f})")
    if r > best:
        best = r
        torch.save(m.user.state_dict(), "user_tower.pt"); torch.save(m.item.state_dict(), "item_tower.pt")
