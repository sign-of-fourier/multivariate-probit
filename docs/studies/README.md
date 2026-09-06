# Studies

Empirical work behind the estimator, one file per study. This is the project's
research archive: what was measured, why the question came up, what it settled,
and what it left open.

The division of labour with the rest of the docs:

- [ifm.md](../ifm.md) holds *mechanism* — the algorithm and its derivations.
- These files hold *measurements*. A derivation keeps only the numbers it needs
  to be legible; the study it came from keeps the rest.
- [limitations.md](../limitations.md) holds *gaps*, linking to whichever study
  established one.

Each study states its question first, gives data provenance rather than data,
records the exact configuration, and ends with what it does **not** establish
and what is still open. Datasets are never committed.

| Study | Question | Status |
| --- | --- | --- |
| [uci-credit.md](uci-credit.md) | Is the composite pairwise objective a legitimate substitute for the full likelihood? | settled for d = 3, moderate positive Σ |
| [margin-gaps.md](margin-gaps.md) | What does `correlation_` estimate when the margins are imperfect? | derivations confirmed; correction still declined |
