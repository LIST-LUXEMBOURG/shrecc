# AGENTS.md for SHRECC development

## Main principles

- Parsimony: every new abstraction, argument, and cache must remove more complexity than it introduces.
- Beginner-first API: preserve the minimal NewDatabase(...).create().write() workflow.
- Explicit behavior: no silent fallback, aggregation, writing, or data substitution.
- Auditability: retain labelled intermediate data and document modelling assumptions.
- Shared implementation: historical and prospective paths should converge on canonical solving, mapping, writing, and LCIA functions.
- Measured optimization: optimize only demonstrated bottlenecks and remove optimizations that do not materially help.
- Local licensed data: never distribute ecoinvent-derived mappings, scores, or caches unless licensing clearly permits it.
- Public API restraint: avoid adding NewDatabase arguments when a result method, internal behavior, or advanced helper is more appropriate.
- Documentation requirement: every public feature must explain what is computed, what is retained in memory, and what is written to Brightway.