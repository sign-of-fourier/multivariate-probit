---
name: Use case
about: Tell us where you are using (or want to use) multivariate-probit
title: "Use case: "
labels: use-case
---

**Domain.** What field or problem? (e.g. fraud, claims, moderation)

**Outcomes.** How many 0/1 outcomes (d), and what are they, roughly?

**Joint query.** Which question do you ask of the model? (`any`, `all`, `none`,
`joint(pattern)`, `conditional`, or something the API does not have yet)

**Size.** Roughly how many rows do you fit on, and how many do you score?

**Inner model.** `linear`, `xgboost`, `rf`, or your own?

**Did the joint result change a decision?** For example a threshold, a review
set, or an action. What changed, compared with treating the outcomes as
independent?

**Anything that got in the way?**
