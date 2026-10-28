import torch, numpy as np
from model import UserTower, ItemTower

torch.manual_seed(42)
D = torch.load("data.pt"); F = D["F"]
PER_DAY, POOL = 100, 14

ut = UserTower(); it = ItemTower()
ut.load_state_dict(torch.load("user_tower.pt"))
it.load_state_dict(torch.load("item_tower.pt"))
ut.eval(); it.eval()

def batch(ix):
    h = D["hist"][ix]; mk = (h >= 0).float().unsqueeze(-1)
    hist = (F[h.clamp(min=0)] * mk).sum(1) / mk.sum(1).clamp(min=1)
    return D["aff"][ix], hist, D["demo"][ix], D["pos"][ix]

va = (D["day"] >= 52).nonzero().squeeze(1)

@torch.no_grad()
def evaluate(k_list=(10, 50, 100)):
    IV = it(F); iday = torch.arange(len(F)) // PER_DAY
    ix = va[torch.randperm(len(va))[:5000]]
    aff, hist, demo, pos = batch(ix)
    day = D["day"][ix][:, None]

    sc = ut(aff, hist, demo) @ IV.T
    ok = (iday[None] <= day) & (iday[None] >= day - POOL)
    sc = sc.masked_fill(~ok, -1e9)

    pos_sc = sc.gather(1, pos[:, None]).squeeze(1)
    ranks  = (sc > pos_sc[:, None]).sum(1)          # 0-indexed rank of the engaged post

    cold = (D["hist"][ix] < 0).all(1); warm = ~cold
    pool_sz = ok.sum(1).float()

    print("=" * 62)
    print("  Two-Tower Recommendation Model  —  Evaluation Report")
    print("=" * 62)
    print(f"\nVal set: days 52-59  |  samples: {len(ix)}  |  avg pool: {pool_sz.mean():.0f} posts")
    rnd = {k: round(k / pool_sz.mean().item(), 3) for k in k_list}
    print(f"Random baseline: " + "  ".join(f"recall@{k}={v}" for k,v in rnd.items()))

    for label, mask in [("ALL", torch.ones(len(ix), dtype=torch.bool)), ("COLD", cold), ("WARM", warm)]:
        r = ranks[mask]; n = mask.sum().item()
        if n == 0: continue
        recalls  = {k: (r < k).float().mean().item() for k in k_list}
        mrr      = (1.0 / (r.float() + 1)).mean().item()
        ndcg10   = ((r < 10).float() / torch.log2(r.float() + 2)).mean().item()
        lift10   = recalls[10]  / rnd[10]
        lift100  = recalls[100] / rnd[100]
        print(f"\n  {label}  (n={n})")
        print(f"    " + "  ".join(f"recall@{k}={recalls[k]:.3f}" for k in k_list))
        print(f"    MRR={mrr:.4f}   NDCG@10={ndcg10:.4f}")
        print(f"    Lift vs random:  @10={lift10:.1f}x   @100={lift100:.1f}x")

    ut_p = sum(p.numel() for p in ut.parameters())
    it_p = sum(p.numel() for p in it.parameters())
    print(f"\n{'='*62}")
    print(f"  UserTower: {ut_p:,} params  |  ItemTower: {it_p:,} params  |  Total: {ut_p+it_p:,}")
    print("=" * 62)

evaluate()
