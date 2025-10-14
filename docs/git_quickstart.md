# Git Quickstart: Committing a Staged File

When `git status` shows a staged file under **"Changes to be committed"**—for example:

```text
$ git status
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   .pip_freeze_311.txt
```

follow these steps to finish the operation:

1. **Review the staged diff**

   ```bash
   git diff --staged
   ```

   Check that the staged content matches what you intend to commit. If something is wrong, unstage it with `git restore --staged <file>` and fix it before continuing.

2. **Create the commit**

   ```bash
   git commit -m "Add pip freeze snapshot for venv311"
   ```

   Use a descriptive message that summarizes the change you are recording.

3. **Push the commit (optional, if you publish to a remote)**

   ```bash
   git push
   ```

   This sends your commit to the remote repository once you are satisfied locally.

These commands complete the workflow required after staging a file so that the new snapshot becomes part of your project history.
