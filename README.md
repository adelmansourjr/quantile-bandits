# When Q-SAR needs a quantile objective

Zhang and Ong’s ICML paper replaces mean ranking with quantile ranking in fixed-budget best-arm identification. I implemented the paper's two-boundary accept/reject algorithm and descending order-statistic convention and held the realised sample budget constant across the allocation and objective controls. The experiment asks when the target quantile justifies the statistical price of estimating it. [Zhang and Ong 2021, Algorithm 1](https://proceedings.mlr.press/v139/zhang21o.html).

Each method selects two of eight stochastic arms by evidence available within the budget. The evaluator knows the analytic lower 20th percentiles. A two-stage budgeted control takes eight samples per arm, then focuses on the four empirical boundary neighbours. All methods share potential reward streams within each of 800 worlds per family. The 320-cap schedule takes 317 draws, also used by every comparator. The [fixed protocol](PROTOCOL.md) defines exponential arms, near-ties and heterogeneous tails before evaluations.

| Family at 317 draws | Q-SAR | Mean-SAR | Uniform quantile | Two-stage quantile |
|---|---:|---:|---:|---:|
| Separated exponential | 34.88% | 64.12% | 30.88% | 24.50% |
| Exponential near ties | 22.75% | 37.50% | 20.25% | 16.25% |
| Heterogeneous Weibull | 47.13% | 14.87% | 41.13% | 40.00% |
| Two-piece tails | 23.50% | 26.12% | 23.50% | 22.75% |

The predeclared Q-SAR–uniform difference in exact set recovery was **+4.00 points** with paired 95% interval **+0.88 to +7.13**. It cleared the threshold. The mean-versus-quantile crossover depended on the generating distributions. In the exponential family, all distributions share shape, so true means and quantiles induce the same order and the mean is a more useful parametric statistic here. Under heterogeneous shapes the objectives diverge and mean-SAR loses. In the two-piece family ordinary quantile estimates struggle near the branch boundary, eliminating the Q-SAR gain over uniform allocation. The heterogeneous and two-piece stress families extend beyond the theorem's hazard assumptions.

A next experiment can test whether feedback coarsening changes the crossover, while keeping amount of label information and sample budget distinct. This run draws independent per-arm rewards.

```sh
python3 run.py --verify results.json
```

Python 3.10+, NumPy 2.x. [Full paired estimates, draw budgets and quantile loss](results.json).
