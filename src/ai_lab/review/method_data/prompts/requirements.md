Requirements: judge the change against EVERY requirement below. Beside `findings`, your JSON object carries `requirements`: one row per requirement, {"id", "verdict", "cites", "evidence"}. verdict: `violated` (the change breaks it), `amended` (the change edits the requirement's own document, consistently with the rest of the change), `met` (the change bears on it and keeps it), `untouched` (the change does not bear on it). cites: the changed files (`path` or `path:line`) the verdict rests on, [] when untouched. evidence: one sentence. A violation is also a finding whose `rule` is that requirement's id.
{{requirement_list}}

{{requirement_documents}}
