"""Compare Q-SAR with exactly matched fixed-budget quantile estimators."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = "da94861198b989dec640c9d5cf9779d280a4ac9c11d91bbf82e499401dfbdcfe"
FAMILIES = ("separated_exponential", "exponential_near_ties", "heterogeneous_weibull", "two_piece")
BUDGETS = 160, 320
TRIALS = 800
K = 8
TOP = 2
TAU = 0.2
METHODS = ("q_sar", "mean_sar", "uniform_quantile", "two_stage_quantile")


def get_world(seed, family):
    rng = np.random.default_rng(seed)
    q = rng.permutation(np.linspace(.17, .58, K))
    q += rng.uniform(-.016, .016, K)
    if family == "exponential_near_ties":
        q = rng.permutation(np.array([.18, .23, .30, .35, .42, .45, .465, .52]))
        q += rng.uniform(-.003, .003, K)
    u = rng.uniform(1e-12, 1-1e-12, (K, max(BUDGETS)))
    if family == "two_piece":
        p = rng.uniform(.15, .25, (K, 1))
        lo = rng.uniform(.04, .22, (K, 1))
        hi = rng.uniform(.62, .95, (K, 1))
        def inverse(prob):
            return np.where(prob < p, lo + .09 * prob / p, hi + .4 * (prob - p) / (1-p))
        return inverse(u), inverse(TAU)[:, 0]
    shape = (rng.uniform(.55, 2.6, (K, 1)) if family == "heterogeneous_weibull"
             else np.ones((K, 1)))
    returns = q[:, None] * (-np.log1p(-u) / -math.log1p(-TAU))**(1 / shape)
    return returns, q


def estimator(samples, mean):
    if mean:
        return float(np.mean(samples))
    # Zhang and Ong use the floor(n*(1-tau))-th descending order statistic.
    rank = max(1, math.floor(len(samples) * (1 - TAU)))
    index = len(samples) - rank
    return float(np.partition(samples, index)[index])


def ranking(arms, payoff, counts, mean):
    return sorted(arms, key=lambda arm: (-estimator(payoff[arm, :counts[arm]], mean), arm))


def sar(payoff, cap, mean=False):
    normalizer = .5 + sum(1 / i for i in range(2, K + 1))
    active = list(range(K))
    committed = []
    counts = np.zeros(K, dtype=int)
    for remaining in range(K, 1, -1):
        n = math.ceil((cap - K) / (normalizer * remaining))
        counts[active] = n
        ranked = ranking(active, payoff, counts, mean)
        need = TOP - len(committed)
        best = ranked[0]
        worst = ranked[-1]
        if need == 0 or need == remaining:
            # The phase schedule stays fixed even if membership is determined.
            commit_best = need == remaining
        else:
            first_below = ranked[need]
            last_above = ranked[need - 1]
            first_gap = estimator(payoff[best, :n], mean) - estimator(payoff[first_below, :n], mean)
            last_gap = estimator(payoff[last_above, :n], mean) - estimator(payoff[worst, :n], mean)
            commit_best = first_gap > last_gap
        if commit_best:
            committed.append(best)
            active.remove(best)
        else:
            active.remove(worst)
    if len(committed) < TOP:
        committed.extend(active)
    assert len(committed) == TOP and int(counts.sum()) <= cap
    return committed, int(counts.sum())


def fixed(payoff, count, two_stage=False):
    if two_stage:
        counts = np.full(K, 8, dtype=int)
        candidate = ranking(range(K), payoff, counts, False)[:4]
        available = count - int(counts.sum())
        equal, extras = divmod(available, len(candidate))
        for i, arm in enumerate(candidate):
            counts[arm] += equal + (i < extras)
    else:
        equal, extras = divmod(count, K)
        counts = np.full(K, equal, dtype=int)
        counts[:extras] += 1
    assert counts.sum() == count
    return ranking(range(K), payoff, counts, False)[:TOP]


def summarize(recommendation, q, exact, losses):
    optimum = np.argsort(q)[-TOP:]
    for name, arms in recommendation.items():
        exact[name].append(set(arms) == set(optimum))
        losses[name].append(float((q[optimum].sum() - q[arms].sum()) / TOP))


def audit():
    for cap in BUDGETS:
        certain = np.repeat(np.arange(K, dtype=float)[:, None], cap, axis=1)
        selected, count = sar(certain, cap)
        assert sorted(selected) == [K-2, K-1] and count <= cap
        assert sorted(fixed(certain, count, True)) == [K-2, K-1]
    assert estimator(np.arange(10), False) == 2


def run():
    audit()
    results = {}
    boot = np.random.default_rng(60553).integers(TRIALS, size=(6000, TRIALS))
    for family_index, family in enumerate(FAMILIES):
        results[family] = {}
        seed_base = 88200 + 1000 * family_index
        for cap in BUDGETS:
            exact = {name: [] for name in METHODS}
            losses = {name: [] for name in METHODS}
            sample_counts = set()
            for seed in range(seed_base, seed_base + TRIALS):
                rewards, q = get_world(seed, family)
                q_set, spent = sar(rewards, cap)
                mean_set, mean_spent = sar(rewards, cap, True)
                assert mean_spent == spent
                sample_counts.add(spent)
                decisions = {"q_sar": q_set, "mean_sar": mean_set,
                             "uniform_quantile": fixed(rewards, spent),
                             "two_stage_quantile": fixed(rewards, spent, True)}
                summarize(decisions, q, exact, losses)
            metrics = {m: {"recovery": float(np.mean(exact[m])), "mean_quantile_loss": float(np.mean(losses[m]))}
                       for m in METHODS}
            contrasts = {}
            for comparator in METHODS[1:]:
                diff = np.array(exact["q_sar"], dtype=int) - exact[comparator]
                contrasts[comparator] = {"recovery_difference": float(diff.mean()),
                                         "interval95": np.percentile(diff[boot].mean(axis=1), [2.5, 97.5]).tolist()}
            results[family][str(cap)] = {"actual_sample_counts": sorted(sample_counts),
                                         "arms": metrics, "q_sar_minus": contrasts}
    primary = results[FAMILIES[0]]["320"]["q_sar_minus"]["uniform_quantile"]
    return {"trials_per_family": TRIALS, "protocol_sha256": PROTOCOL, "primary": primary,
            "success_gate": primary["recovery_difference"] >= .02 and primary["interval95"][0] > 0,
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--write", type=Path)
    outputs.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest() != PROTOCOL:
        raise ValueError("Local protocol changed")
    data = run()
    if args.verify:
        if data != json.loads(args.verify.read_text()):
            raise SystemExit("Results did not replay exactly")
        print("Replayed all 6,400 synthetic trials and verified their budget invariants.")
    elif args.write:
        args.write.write_text(json.dumps(data, indent=2) + "\n")
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
