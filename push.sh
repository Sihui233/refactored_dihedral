#!/usr/bin/env bash
# push.sh — one-command branch checkout/creation, commit, and push
# Usage: ./push.sh <branch> "<commit message>"

set -e  # exit immediately if any command fails

# --- Argument check ------------------------------------------------------
if [ $# -lt 2 ]; then
  echo "Usage: $0 <branch> \"<commit message>\""
  exit 1
fi

BRANCH="$1"
MSG="$2"

# --- 1. Switch to the target branch (create it if it doesn’t exist) ------
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "Switching to existing branch $BRANCH"
  git checkout "$BRANCH"
else
  echo "Creating and switching to new branch $BRANCH"
  git checkout -b "$BRANCH"
fi

# --- 2. Stage and commit -------------------------------------------------
# Commit only if there are changes, to avoid empty commits
if ! git diff --quiet || ! git diff --cached --quiet; then
  git add .
  git commit -m "$MSG"
  echo "Committed: $MSG"
else
  echo "No changes detected—skipping commit."
fi

# --- 3. Push to remote ---------------------------------------------------
git push -u origin "$BRANCH"
echo "Pushed to origin/$BRANCH and set upstream."
