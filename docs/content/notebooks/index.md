# Notebooks

The two notebooks below are the validated public examples for the harmonized
`NewDatabase` workflow.

Start with `1_shrecc_get_started` for the shortest complete path from a
historical Energy Charts configuration to a written Brightway database.

The expanded `2_shrecc_analysis` notebook uses one `NewDatabase` object across
historical Energy Charts and prospective TYNDP years. It explores Brightway
preflight, normalized mixes, physical volumes, activity mapping, domestic and
imported wind, fallback diagnostics, cutoff tables, and the separate
database-writing step. Multi-year writes create one year-labelled foreground
database per modeled year.

The former source-specific notebooks remain in `notebooks/archive/` as a record
of the development process, but they are no longer part of the public workflow.

```{toctree}
---
maxdepth: 2
caption: Contents
---
1_shrecc_get_started
2_shrecc_analysis
```
