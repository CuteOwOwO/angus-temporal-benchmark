# Generation provenance

- `scaled_v1/`: unique-category composition and QA generation.
- `repeated_v2/`: occurrence-aware repeated-category composition and QA generation.
- `final/`: final benchmark aggregation and release validation.

These are frozen provenance copies. The portable `benchmark/` directory does
not need them at evaluation time. Full regeneration of the parent releases
also requires the reviewed source pools retained in the original workspace.
