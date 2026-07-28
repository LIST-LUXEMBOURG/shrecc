# Notebooks

The four notebooks below are the validated public examples for the harmonized
`NewDatabase` workflow.

Start with `1_shrecc_get_started` for the shortest complete path from a
historical Energy Charts configuration to a written Brightway database.

The expanded `2_shrecc_analysis` notebook uses one `NewDatabase` object across
historical Energy Charts and prospective TYNDP years. It explores Brightway
preflight, normalized mixes, physical volumes, activity mapping, domestic and
imported wind, fallback diagnostics, cutoff tables, and the separate
database-writing step. Multi-year writes create one year-labelled foreground
database per modeled year.

`3_shrecc_annual_validation` compares annual SHRECC results with
ecoinvent 3.11 for 2021 and premise-modified ecoinvent using REMIND-EU's
SSP2-PkBudg650 scenario for 2035, 2040, and 2050. The comparison highlights
several interpretation and development points:

- The SHRECC annual mixes are weighted by measured national demand to match the
  implicit volume weighting of annual ecoinvent and premise inventories.
- Country-specific discrepancies remain. For example, ecoinvent uses
  market-based data for Switzerland, while Energy Charts has limited coverage
  for the Netherlands.
- Differences in background-database activity mapping can also affect the
  comparison.

Finally, `4_shrecc_consumption_profiles` compares flat, national-demand, and
custom ELMAS consumption profiles. It resolves each foreground and background
low-voltage inventory to generating technologies and compares their
climate-change scores.

The former source-specific notebooks remain in `notebooks/archive/` as a record
of the development process, but they are no longer part of the public workflow.

```{toctree}
---
maxdepth: 2
caption: Contents
---
1_shrecc_get_started
2_shrecc_analysis
3_shrecc_annual_validation
4_shrecc_consumption_profiles
```
