# AGENTS.md for SHRECC development

## Main principles

- Parsimony: every new abstraction, argument, and cache must remove more complexity than it introduces. Avoid new dependencies, and keep minimal diffs.
- Beginner-first API: preserve the minimal NewDatabase(...).create().write() workflow.
- Explicit behavior: no silent fallback, aggregation, writing, or data substitution.
- Auditability: retain labelled intermediate data and document modelling assumptions.
- Shared implementation: historical and prospective paths should converge on canonical solving, mapping, writing, and LCIA functions.
- Measured optimization: optimize only demonstrated bottlenecks and remove optimizations that do not materially help.
- Local licensed data: never distribute ecoinvent-derived mappings, scores, or caches unless licensing clearly permits it.
- Public API restraint: avoid adding NewDatabase arguments when a result method, internal behavior, or advanced helper is more appropriate.
- Documentation requirement: every public feature must explain what is computed, what is retained in memory, and what is written to Brightway.

## Must-read sources

- [Implementation plan](docs/implementation-plan.md): decisions, current work,
  and next steps.
- [Architecture](docs/architecture.md): module boundaries and dependency flow.
- [Data and outputs](docs/content/data.md): user-facing data structures and
  behavior published in the documentation.

## Response style

- Do not explain basic Python concepts.

## Testing

- Generate focused tests only for modified behavior.
- Avoid redundant test scaffolding.

## Token efficiency

- Do not repeat repository context.
- Avoid reprinting unchanged code.
- Prefer concise bullet summaries over long prose.
- Ask for missing files instead of assuming implementations.
