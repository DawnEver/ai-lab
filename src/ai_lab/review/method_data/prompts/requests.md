A turn may answer with `requests` instead of `findings`, and the next turn carries their results:
- `diff`: [paths] -- the change to those paths;
- `show`: [paths] -- the file as it is at the head;
- `grep`: [patterns] -- `git grep -n` over the head, to follow a call or a name into unchanged code;
- `list`: [directories] -- the changed files under a directory.
You have {{turns}} turns; the last must answer with `findings`.
