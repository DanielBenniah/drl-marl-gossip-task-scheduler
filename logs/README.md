# Run logs (audit trail)

These text files capture stdout from long experiment runs. They are kept for
reproducibility and debugging.

**Path note:** Some log lines still mention directories like `results_g2_*` or
`results_adaptive_*` at the repository root. The committed outputs were later
consolidated under `results/` with shorter names (for example
`results_g2_table2_lam05` → `results/g2_table2_lam05`). Treat the CSV and JSON
under `results/` as the canonical artifacts; treat these logs as historical
console output.
