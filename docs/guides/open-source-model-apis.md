# Open-Source Model APIs

For sustained research, Praxist generally favors APIs serving capable
open-source or open-weight models when a representative run demonstrates high
cache reuse, sufficient research quality, and stable throughput. This is a
selection policy, not a hidden runtime default: the operator still chooses the
profile during setup.

## Current Shortlist

| Priority | Option | Selection note |
|---|---|---|
| 1 | **DeepSeek V4 Pro** | Praxist provides a maintained direct API profile. Evaluate it first where the service is available and appropriate for the project. |
| 2 | **Open-source models through OpenRouter** | Use the OpenRouter profile when routing flexibility matters. Select the exact model explicitly and verify that its route reports useful cache reuse. |
| 3 | **Open-source models through OrcaRouter** | Use the OrcaRouter profile when adaptive routing and zero-markup inference matter. The `orcarouter/auto` model picks a live route per request. |
| 4 | **Operator-managed open-source model endpoints** | Add or select a compatible provider plugin when deployment policy requires a private or self-hosted endpoint. Validate the plugin contract before a long run. |

The order is a practical starting point, not a claim that one model is best for
every task. Availability, pricing, model behavior, and provider-side caching can
change independently.

## Validate Before a Long Run

1. Run a short, representative workload through the intended agent runtime and
   API route.
2. Confirm output quality against the task's normal evaluator rather than a
   provider-specific proxy.
3. Inspect cached and uncached input usage, latency, and failures in the run
   artifacts.
4. Keep the selected route only when the total cost and research quality are
   acceptable together.

Praxist preserves stable prompt prefixes where the selected route supports
them, but no model name alone guarantees a high cache-hit rate. See
[Cost Optimization](cost-optimization.md) for the cache contract and
[API Providers](model-providers.md) for supported provider shapes.
