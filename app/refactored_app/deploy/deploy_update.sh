#!/usr/bin/env bash
set -euo pipefail

# Deploy a new release zip by switching /home/pi/refactored_app_current -> new release

SERVICE_NAME="streamlit-app"
RELEASES_DIR="/home/pi/releases"
SYMLINK="/home/pi/refactored_app_current"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/release.zip" >&2
  exit 2
fi

ZIP="$1"
if [[ ! -f "$ZIP" ]]; then
  echo "zip not found: $ZIP" >&2
  exit 2
fi

ts=$(date +%Y%m%d_%H%M%S)
dest="$RELEASES_DIR/refactored_app_$ts"

sudo systemctl stop "$SERVICE_NAME"
mkdir -p "$dest"
unzip -q "$ZIP" -d "$dest"

# Expect a top-level folder named "refactored_app" inside the zip.
ln -sfn "$dest/refactored_app" "$SYMLINK"

sudo systemctl start "$SERVICE_NAME"
sudo systemctl status "$SERVICE_NAME" --no-pager