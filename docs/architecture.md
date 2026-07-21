# Architecture

SHRECC separates source data, network algebra, background-database mapping,
and Brightway writing. The canonical dependency flow is:

```text
Energy Charts or TYNDP
        |
        v
production_volume + trade_volume + optional consumption_volume
        |
        v
shared country-network solver
        |
        v
consumption_mix + retained physical volumes
        |
        v
ecoinvent or premise mapping
        |
        v
filtered activity table
        |
        v
Brightway database
```

## Modules

- `energy_charts.py`: Energy Charts API access, cleaning, canonical input
  construction, reduced solving, and source-specific cache orchestration.
- `tyndp.py`: TYNDP acquisition, scenario construction, canonical input
  construction, reduced solving, and source-specific cache orchestration.
- `solver.py`: Source-independent country-network algebra and canonical result
  variables.
- `result_store.py`: Versioned, compressed time-chunk persistence and selective
  loading.
- `mapping.py`: Shared time filtering, ecoinvent activity allocation, fallback
  handling, and database-table conversion.
- `premise_mapping.py`: Prospective activity and IAM-region mapping built on the
  shared mapping conventions.
- `database.py`: Cutoff handling, Brightway activity lookup, and database
  construction/writing.
- `pipeline.py`: The public `NewDatabase` workflow and inspectable pipeline
  state.
- `treatment.py`: Temporary legacy full-matrix implementation retained for
  validation. New code should not depend on it.

`download.py` and `activity_mapping.py` are compatibility facades for older
imports. They contain no independent implementation and can be removed during a
future breaking cleanup.

## Design Rules

1. Source modules end at canonical solver inputs or canonical results.
2. The solver has no knowledge of Energy Charts, TYNDP, ecoinvent, premise, or
   Brightway.
3. Mapping depends on canonical results, never on source-specific raw tables.
4. Database writing depends on mapped activity tables, never on graph algebra.
5. High-level orchestration belongs in `pipeline.py` and must compose the lower
   layers rather than reimplement them.
