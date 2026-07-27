# Mapping source files

This directory contains editable source material used to maintain SHRECC's
runtime mapping tables. Files here are version-controlled project resources but
are not included in the installed Python package.

- `electricity_sources.xlsx` is the authoring concordance between electricity
  activity/resource names and Energy Charts or NED.nl production categories.
- The generated country-specific allocation tables used at runtime live under
  `shrecc/data/`.

Keeping source concordances separate from generated runtime tables makes the
package contents explicit without losing the information needed to update those
tables.
