# AI-B-roll placeholder

This directory is left out of version control to keep the main repository lightweight.

## Populate locally
1. Clone or copy your AI B-roll helper project into `AI-B-roll/`.
2. Add your b-roll assets under `AI-B-roll/broll_library/` (the folder stays ignored).
3. Keep any API keys or models outside the Git repo.

## Optional: re-create the former submodule
```
git submodule add <URL_DU_REPO_BROLL> AI-B-roll
git submodule update --init --recursive
```
Replace `<URL_DU_REPO_BROLL>` with the repository that hosts your AI b-roll logic.

