---
name: keys-live-in-the-git-credential-store
description: provider keys are read only from git credential (host of the endpoint, username api-key); ai_lab.auth stores them; env vars retired
created: 2026-10-13
---
# Keys live in the git credential store

- User ruling 2026-10-13: one key mechanism on Windows and Linux. `Provider.key_name` replaces `key_env`
  (a user row still naming `key_env` is refused); the key is `lab_commons.credentials.credential_for(host)`.
- `python -m ai_lab.auth login <provider>` (getpass/stdin), `--from-file <path|->` (NAME=value,
  export NAME=value, NAME value), `status` (stored/missing, never the secret). Providers on one host share a key.
- Branch `feat/credential-keys` 75843ed, unpushed; needs lab-commons `feat/credentials-runtime` on GitHub.
- 5 architecture tests were red on main before this branch (shared-registry IDs and family config drift).
