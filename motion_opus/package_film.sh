#!/bin/sh
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out="$base/film_portable"
mkdir -p "$out"
cp "$base/index.html" "$base/README.md" "$base/sources.json" "$base/STORYBOARD.md" "$base/SPEECH.md" "$out/"
cp -R "$base/assets" "$base/data" "$out/"
if find "$out" -type l | grep -q .; then
  echo "Packaging failed: symlink found" >&2
  exit 1
fi
if grep -E 'https?://' "$out/index.html" >/dev/null; then
  echo "Packaging failed: network URL in film" >&2
  exit 1
fi
test -s "$out/index.html"
echo "Portable film ready: $out"
