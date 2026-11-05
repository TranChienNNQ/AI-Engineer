"""Lesson 4: probability, statistics and uncertainty with synthetic examples.

NumPy only. Run python3 lab.py --mode all.
Random experiments use independent Bernoulli observations with a fixed p.
"""

import argparse
import math

import numpy as np


SEED = 2026


def probability(p):
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Probability must be finite and between 0 and 1.")
    return p


def binomial_pmf(k, n, p):
    probability(p)
    if n < 0 or int(n) != n or int(k) != k or not 0 <= k <= n:
        raise ValueError("Use integer n >= 0 and 0 <= k <= n.")
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def bayes_error_given_alarm(prior, true_alarm_rate, false_alarm_rate):
    """P(error | alarm), given P(error), P(alarm|error), P(alarm|no error)."""
    for value in (prior, true_alarm_rate, false_alarm_rate):
        probability(value)
    numerator = prior * true_alarm_rate
    alarm_rate = numerator + (1 - prior) * false_alarm_rate
    if alarm_rate == 0:
        raise ValueError("P(alarm)=0: conditional probability is undefined.")
    return numerator / alarm_rate


def descriptive_stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)):
        raise ValueError("Use at least two finite scalar observations.")
    return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
            "var_ddof0": float(np.var(a, ddof=0)),
            "var_ddof1": float(np.var(a, ddof=1)),
            "std_ddof0": float(np.std(a, ddof=0)),
            "std_ddof1": float(np.std(a, ddof=1))}


def sample_proportions(p, n, repeats=5000, seed=SEED):
    """Each count is a new experiment of n iid Bernoulli(p) observations."""
    probability(p)
    rng = np.random.default_rng(seed)
    counts = rng.binomial(n, p, size=repeats)
    return counts / n


def wilson_interval(k, n, z=1.96):
    """Two-sided Wilson score interval; z=1.96 targets nominal 95% coverage."""
    if int(n) != n or int(k) != k or n <= 0 or not 0 <= k <= n:
        raise ValueError("Use integer n > 0 and 0 <= k <= n.")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive.")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def exact_wilson_coverage(p, n):
    """Enumerate the binomial model, for small teaching examples only."""
    return sum(binomial_pmf(k, n, p) for k in range(n + 1)
               if wilson_interval(k, n)[0] <= p <= wilson_interval(k, n)[1])


def binary_entropy(p):
    """Entropy in bits; define 0 * log2(0) = 0 by continuity."""
    probability(p)
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def probability_scores(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not np.all((y == 0) | (y == 1)):
        raise ValueError("Use matching 1-D binary targets and probabilities.")
    if not len(y) or not np.all(np.isfinite(p)) or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Use nonempty finite probabilities in [0,1].")
    brier = float(np.mean((p - y) ** 2))
    # A contradictory p=0 or p=1 has infinite log loss, not an arbitrary finite clip.
    with np.errstate(divide="ignore"):
        log_loss = float(-np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))
    accuracy = float(np.mean((p >= 0.5) == y))
    return accuracy, brier, log_loss


def calibration_example():
    """Three constructed groups, ten observations each; not a fitted model."""
    y = np.concatenate([np.r_[np.ones(k), np.zeros(10 - k)] for k in (2, 5, 8)])
    p_a = np.repeat([0.2, 0.5, 0.8], 10)
    p_b = np.repeat([0.05, 0.5, 0.95], 10)
    return y, p_a, p_b


def probability_demo():
    p, n = 0.2, 10
    print("Bernoulli p=0.2: E[X]=0.2; Var(X)=0.16")
    print(f"P(K=2), K~Binomial(n=10,p=0.2): {binomial_pmf(2,n,p):.9f}")
    print(f"Sum of binomial probabilities: {sum(binomial_pmf(k,n,p) for k in range(n+1)):.9f}")


def stats_demo():
    for name, values in [("steady", [20, 20, 20, 20, 20]),
                          ("tail", [10, 10, 10, 10, 60])]:
        print(name, descriptive_stats(values))


def bayes_demo():
    print("Per 10000 requests: error=100, no error=9900")
    print("Alarms: true=90, false=495, total=585")
    for prior in (0.01, 0.10):
        result=bayes_error_given_alarm(prior,0.9,0.05)
        print(f"P(error)={prior:.2f} -> P(error|alarm)={result:.6f}")


def sampling_demo(repeats, seed):
    for n in (100, 1000):
        estimates=sample_proportions(.2,n,repeats,seed)
        theory=math.sqrt(.2*.8/n)
        print(f"n={n}; repeats={repeats}; mean(phat)={np.mean(estimates):.6f}; "
              f"empirical SD={np.std(estimates,ddof=1):.6f}; theoretical SE={theory:.6f}")


def intervals_demo(repeats, seed):
    for k,n in [(20,100),(200,1000),(0,100)]:
        lo,hi=wilson_interval(k,n)
        print(f"k={k}, n={n}, phat={k/n:.3f}: Wilson [{lo:.6f}, {hi:.6f}]")
    counts=np.random.default_rng(seed).binomial(100,.2,size=repeats)
    covered=[lo <= .2 <= hi for lo,hi in (wilson_interval(int(k),100) for k in counts)]
    print(f"Simulated coverage: {np.mean(covered):.6f} ({repeats} experiments)")
    print(f"Exact coverage under Binomial(100,0.2): {exact_wilson_coverage(.2,100):.6f}")


def calibration_demo():
    y,p_a,p_b=calibration_example()
    for name,p in [("A",p_a),("B",p_b)]:
        accuracy,brier,logloss=probability_scores(y,p)
        print(f"Model {name}: accuracy={accuracy:.6f}; Brier={brier:.6f}; log loss={logloss:.6f}")
    for p in (0.0,0.05,0.2,0.5,0.8,0.95,1.0):
        print(f"H({p:g})={binary_entropy(p):.6f} bits")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["all","probability","stats","bayes","sampling","intervals","calibration"],default="all")
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--repeats",type=int,default=5000)
    args=parser.parse_args()
    if args.seed < 0 or args.repeats < 2:
        parser.error("Use seed >= 0 and repeats >= 2.")
    demos={"probability":probability_demo,"stats":stats_demo,"bayes":bayes_demo,
           "sampling":lambda:sampling_demo(args.repeats,args.seed),
           "intervals":lambda:intervals_demo(args.repeats,args.seed),"calibration":calibration_demo}
    for name,demo in demos.items():
        if args.mode in ("all",name):
            print(f"\n--- {name} ---")
            demo()


if __name__=="__main__":
    main()
