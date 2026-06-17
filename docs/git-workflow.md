# Git & GitHub Workflow

Everything flows through one rule: **feature branches → dev → main**.

```
main        ← stable, reviewed code only. Never commit here directly.
  └── dev   ← integration branch. All features merge here first.
        └── feature/issue-N-description  ← one branch per issue
```

---

## Branch Structure

| Branch | Purpose | Who pushes |
|--------|---------|------------|
| `main` | Stable, paper-ready code | Merged via PR from `dev` only |
| `dev` | Integration — all features merge here | Both members (via PR merge) |
| `feature/issue-N-*` | One issue, one branch | The assignee of that issue |

---

## Daily Workflow (step by step)

### 1. Pick up a task

```bash
# Open Claude Code in the project root
claude

# Then type:
/start-task
```

Claude lists your assigned issues, you pick one, and it handles everything:
- Moves the issue to **Ready** on the board
- Creates `feature/issue-N-description` branching off `dev`
- Moves the issue to **In Progress**

Or jump straight to a specific issue:
```bash
/start-task 4
```

### 2. Implement on your feature branch

Work entirely on your `feature/issue-N-*` branch. Commit often:

```bash
git add src/attention.py          # always name specific files
git commit -m "issue #1: add scaled dot-product attention"
git add src/attention.py
git commit -m "issue #1: add multi-head projection and output layer"
```

**Commit message format:** `issue #N: what this commit does`
This auto-links the commit to the GitHub issue.

### 3. Test your work

When done implementing, run your code and check all acceptance criteria in the issue before merging.

### 4. Merge to dev (no PR needed for this step)

Once everything works, `/start-task` does this automatically. Or manually:

```bash
git checkout dev
git pull origin dev                         # get latest from teammate
git merge --no-ff feature/issue-1-mha -m "issue #1: merge into dev"
git push origin dev
```

The `--no-ff` flag keeps the merge as a visible commit in history.

### 5. Raise a PR from dev → main

`/start-task` raises this automatically at the end. Or manually:

```bash
gh pr create \
  --base main \
  --head dev \
  --title "issue #1: Implement vanilla multi-head attention" \
  --body "Closes #1"
```

The other team member reviews and merges.

---

## Handling Conflicts

If `dev` was updated by your teammate while you were working:

```bash
git checkout dev
git pull origin dev
git checkout feature/issue-N-your-branch
git merge dev                   # bring dev changes into your feature branch
# resolve any conflicts, then:
git add conflicting_file.py
git merge --continue
```

Merge `dev` into your feature branch (not the other way around) to keep `dev` clean.

---

## Quick Reference

```bash
# Start a new task
/start-task [issue-number]

# Check what branch you're on
git branch

# See what's changed
git status
git diff

# See recent history
git log --oneline -10

# Update your feature branch with latest dev
git merge dev

# Push your feature branch (for backup / sharing mid-task)
git push origin feature/issue-N-your-branch

# See all remote branches
git branch -r
```

---

## Project Board States

Issues move through these states automatically when you use `/start-task`:

```
Backlog → Ready → In Progress → In Review → Done
```

| State | Meaning |
|-------|---------|
| Backlog | Not started yet |
| Ready | Picked up, branch created |
| In Progress | Actively being worked on |
| In Review | PR raised, waiting for review |
| Done | Merged to main |

Board: **https://github.com/users/YUVARAJ-R-ai/projects/5**

---

## Rules

1. **Never commit directly to `main` or `dev`** — always use a feature branch
2. **One issue = one feature branch** — don't mix issues on the same branch
3. **Always pull `dev` before branching** — avoids conflicts
4. **Commit message must include `issue #N`** — auto-links to GitHub
5. **PR from dev → main requires the other person to review** — don't self-merge
