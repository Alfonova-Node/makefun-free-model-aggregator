# The Free Shelf — a free-model aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Alfonova-Node/makefun-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Alfonova-Node/makefun-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A dual-wire aggregator proxying OpenAI, Anthropic, and Gemini free tiers, with best-free-model routing and budget-aware failover. It serves both `/v1/models` shapes and prints a byte-identical registered catalog with `--list`.

## Run it

```sh
node main.ts --list
```

The trial environment supplies the provider directory, workload, and endpoints that drive the server.

<sub>Entered on [make.fun](https://make.fun/?ref=r4jyg5rt&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
