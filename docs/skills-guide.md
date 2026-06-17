# Claude Code Skills Guide

This repo ships four Claude Code slash commands inside `.claude/skills/`.
They automate the entire GitHub workflow — no browser needed for issue tracking or branch setup.

## Prerequisites

1. **Install Claude Code**
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Authenticate GitHub CLI**
   ```bash
   gh auth login
   # choose: GitHub.com → SSH → browser login
   ```

3. **Open the project in Claude Code**
   ```bash
   cd /path/to/torrent_llm
   claude
   ```
   The skills in `.claude/skills/` are picked up automatically.

---

## Skill Reference

### `/new-issue` — Log a task or bug

Use this whenever you think of something that needs doing but aren't starting it right now.

```
/new-issue "implement DHT node discovery via Hivemind"
```

What it does:
- Detects the issue type (feature / bug / chore / docs)
- Generates a user story with acceptance criteria for you to confirm
- Creates the GitHub issue and adds it to the **Backlog** on the project board

Options:
```
/new-issue                    # fully interactive — it asks you for the description
/new-issue "description"      # seed description, skips the opening prompt
```

---

### `/start-task` — Pick up and implement your next assigned task

Use this when you're ready to start coding.

```
/start-task
```

What it does:
1. Lists your assigned issues from the board
2. Refines the selected issue into an implementation-ready user story
3. Moves it to **Ready** → creates a feature branch → moves it to **In Progress**
4. Guides you through implementing, committing, and testing
5. Merges into `dev`, raises a PR to `main`, moves issue to **In Review**

Options:
```
/start-task 6          # jump straight to issue #6
/start-task --pr-only  # only raise the PR for what's already on dev
```

Your assigned Systems issues are: **#3 #4 #5 #6** (Sprint 1–2) and **#15 #16 #17 #19** (Backlog).
Start with Sprint 1: **#3** (test rig setup).

---

### `/project-research` — Deep-research a project idea

Use this at the start of a new project or sub-project to generate a structured research brief.

```
/project-research
```

What it does:
- Searches the web and GitHub for competitors, prior work, and relevant libraries
- Categorises features into Core / Important / Nice-to-have / Don't Build
- Breaks every feature into concrete dev tasks with size estimates
- Writes everything to `docs/research.md`

---

### `/plan-project` — Manage the GitHub Project board

Use this to update sprints, add new issues, or reassign tasks.

```
/plan-project          # full planning flow
/plan-project sprint   # plan the next sprint from existing backlog
/plan-project assign   # reassign issues to teammates
```

---

## Typical Daily Workflow

```
# Start your day
/start-task            → picks up your next assigned issue, branches, you implement

# Found a bug or new idea mid-day
/new-issue "description"   → logs it to Backlog without interrupting current work

# Ready to submit work for review
# (start-task handles the PR automatically — just say "ok" when tests pass)
```

---

## Project Board

All issues are visible at:
**https://github.com/users/YUVARAJ-R-ai/projects/5**

Columns: **Backlog → Ready → In Progress → In Review → Done**

| Sprint | Due | Your issues (Systems) |
|--------|-----|-----------------------|
| Sprint 1 — Foundations | Jun 20 | #3 test rig |
| Sprint 2 — Baselines | Jun 27 | #4 layer split, #5 gRPC transport, #6 profiler |
| Backlog (stretch) | — | #15 DHT, #16 reroute, #17 heterogeneous routing, #19 credit layer |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gh: authentication required` | Run `gh auth login` |
| `Not inside a git repo` | `cd` into the project root before running `claude` |
| Skill not found | Make sure you're in the `torrent_llm` directory; `.claude/` must be in the project root |
| Board not found | Check `gh project list --owner YUVARAJ-R-ai` — project is named `Torrent_llm` (#5) |
