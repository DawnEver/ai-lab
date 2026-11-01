# ai-lab

Typed decisions and schema-bound responses from any model, through one interface (`decide`, `respond`).

## How a change lands

The family flow, stated once in [the integration branch](https://github.com/DawnEver/lab-commons/blob/integrate/main/docs-src/dev/integration.md):

- Cut `pr/<topic>` from `origin/integrate/main`, measure what you touched, commit, push, open a pull request.
- A producer claims the pushed head and posts `tests/<part>`; the FORGE merges the PR once every part
  stands PASS, the head merges cleanly and the merge passes the audit.
- `main` moves only by `integrate promote <sha>` on a full PASS.

The mechanisms are dev-lab's (`docs-src/dev/`); this repo declares the names in `pyproject.toml`.
