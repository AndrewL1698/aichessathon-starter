#!/usr/bin/env bash
# Download the Sunfish engine (github.com/thomasahle/sunfish, GPL-3) for the local opponent in
# local-opponents/sunfish. It is not vendored: this repo ships no engine it did not write.
set -euo pipefail

COMMIT=436f2d18dc2396b623928f4b878ba7c97c964cca
SHA256=5d9ba12a615e6394b634b22ca2ad33fd4720bb6ca38e327717be7891060e2819
URL="https://raw.githubusercontent.com/thomasahle/sunfish/${COMMIT}/sunfish.py"
DESTINATION="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sunfish/sunfish.py"

scratch=$(mktemp)
trap 'rm -f "$scratch"' EXIT
curl -fsSL "$URL" -o "$scratch"

# coreutils on Linux, the BSD spelling on macOS.
if command -v sha256sum > /dev/null; then
  got=$(sha256sum "$scratch" | cut -d' ' -f1)
else
  got=$(shasum -a 256 "$scratch" | cut -d' ' -f1)
fi
if [ "$got" != "$SHA256" ]; then
  echo "sha256 mismatch for $URL: expected $SHA256, got $got" >&2
  exit 1
fi

mv "$scratch" "$DESTINATION"
trap - EXIT
# mktemp makes the file private to us; it is source everyone reads.
chmod 644 "$DESTINATION"
echo "Wrote $DESTINATION at $COMMIT"
