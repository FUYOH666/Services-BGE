# Canon

Shared facts for every agent of Kestrel Works. Read this file in one pass. The planner curates it; other agents propose changes as files in `inbox/`.

## Operator

- Mira Okafor runs Kestrel Works, a two-person studio that automates invoice and contract checks for small logistics firms.
- Working languages: English and Russian.

## How the team works

- HQ and curator: `planner`. It files tasks, reviews hand-backs and edits this file.
- Code and repositories: `coder`. Research: `scout`. Outreach and follow-ups: `herald`. Independent review: `critic`, which runs on a different model family.
- Tasks are issues in the `kestrel/ops` repository with a `to:<agent>` label.

## Standing rules

- Nothing is sent in Mira's name (email, post, payment) without her written "yes" for that action.
- No secrets, tokens or machine addresses in git.
- The production invoice checker is `ledgerlens` on branch `stable`. `ledgerlens-next` is a prototype and never runs on client data.
- Backups of the document store run nightly at 02:30 and are kept for 30 days.
