#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

release_base_url="${RNA_WORKBENCH_RELEASE_BASE_URL:-https://github.com/quantumtative/GoForth/releases/download/v0.1.0}"
checkpoint_dir="${RNA_WORKBENCH_CHECKPOINT_DIR:-checkpoints}"

mkdir -p "$checkpoint_dir"

download() {
  local filename="$1"
  local expected_sha="$2"
  local output="$checkpoint_dir/$filename"
  local url="${release_base_url%/}/$filename"

  if [[ -f "$output" ]]; then
    echo "Found $output"
  else
    echo "Downloading $url"
    curl --fail --location --output "$output" "$url"
  fi

  local actual_sha
  actual_sha="$(shasum -a 256 "$output" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "Checksum mismatch for $output" >&2
    echo "expected: $expected_sha" >&2
    echo "actual:   $actual_sha" >&2
    exit 1
  fi
  echo "Verified $output"
}

download "full_structure_small.pt" "a28e650ba0a8fd61a92ade424d939f3d95631df63f6f9f2ae78a5f81932b472f"
download "fsb_partial_base_small.pt" "38e7a884c26976e003c50fac290d7bf8d636fa13c645a348f47f9fe86a3de09d"

echo "Checkpoints are ready in $checkpoint_dir"
