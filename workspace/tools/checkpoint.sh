#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[checkpoint] git repo not initialized at $ROOT" >&2
  exit 1
fi

max_points=${CHECKPOINT_MAX:-30}
prefix=${CHECKPOINT_PREFIX:-checkpoint-}
message=${1:-""}

last_tag=$(git tag -l "${prefix}*" | sed -E "s/${prefix}//" | grep -E '^[0-9]+$' | sort -n | tail -n1 || true)
if [[ -z "$last_tag" ]]; then
  next_idx=1
else
  next_idx=$((last_tag + 1))
fi

tag_name="${prefix}$(printf "%03d" "$next_idx")"
commit_msg=${message:-"${tag_name}"}

git add -A
git commit -m "$commit_msg" || echo "[checkpoint] nothing to commit, tagging anyway"

current_commit=$(git rev-parse HEAD)
if git rev-parse "$tag_name" >/dev/null 2>&1; then
  echo "[checkpoint] tag $tag_name already exists" >&2
  exit 1
fi

git tag "$tag_name" "$current_commit"

tags_to_prune=$(git tag -l "${prefix}*" | sed -E "s/${prefix}//" | grep -E '^[0-9]+$' | sort -n)
count=$(echo "$tags_to_prune" | wc -l)
if (( count > max_points )); then
  to_delete=$(echo "$tags_to_prune" | head -n $((count - max_points)))
  while read -r idx; do
    [[ -z "$idx" ]] && continue
    git tag -d "${prefix}${idx}" >/dev/null 2>&1 || true
  done <<< "$to_delete"
fi

echo "[checkpoint] created $tag_name at $current_commit"
