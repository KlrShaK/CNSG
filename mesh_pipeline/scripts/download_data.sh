#!/usr/bin/env bash
set -euo pipefail

# Resolve important paths relative to this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
DATA_DIR="$ROOT_DIR/data"

ZIP_URL="https://cvg-data.inf.ethz.ch/lamar/raw/HGE/sessions/navvis_2022-02-06_12.55.11.zip"
ZIP_NAME="$(basename "$ZIP_URL")"
ZIP_PATH="$DATA_DIR/$ZIP_NAME"

# Google Drive folder ID extracted from the provided link
GDRIVE_FOLDER_ID="1x01Lll0zOZ78TFRxLhJrw9z6GU94FaCW"

mkdir -p "$DATA_DIR"

echo "Downloading $ZIP_NAME ..."
curl -fL --retry 3 "$ZIP_URL" -o "$ZIP_PATH"

echo "Unpacking $ZIP_NAME into $DATA_DIR ..."
unzip -o "$ZIP_PATH" -d "$DATA_DIR"

echo "Removing downloaded archive ..."
rm -f "$ZIP_PATH"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found, attempting installation via pip (user scope) ..."
  if ! python3 -m pip install --user gdown; then
    echo "Failed to install gdown automatically. Please install it manually and re-run the script." >&2
    exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Downloading Google Drive folder into $DATA_DIR ..."
gdown --folder --id "$GDRIVE_FOLDER_ID" --output "$DATA_DIR"

# Handle localization.zip if it exists
LOCALIZATION_ZIP="$(find "$DATA_DIR" -maxdepth 2 -type f -name "localization.zip" | head -n 1 || true)"

if [[ -n "$LOCALIZATION_ZIP" && -f "$LOCALIZATION_ZIP" ]]; then
  echo "Found localization.zip: $LOCALIZATION_ZIP"
  echo "Unzipping to $ROOT_DIR ..."
  unzip -o "$LOCALIZATION_ZIP" -d "$ROOT_DIR"
  echo "Localization outputs extracted to $ROOT_DIR/outputs"
  rm -f "$LOCALIZATION_ZIP"
else
  echo "No localization.zip found. Skipping localization outputs."
fi

# Handle additional processed data zip (depth_maps + semantic_masks)
SESSION_DIR="$DATA_DIR/navvis_2022-02-06_12.55.11"
ADDITIONAL_ZIP="$(find "$DATA_DIR" -maxdepth 2 -type f -name "additional_data_navvis_2022-02-06_12.55.11.zip" | head -n 1 || true)"

if [[ -n "$ADDITIONAL_ZIP" && -f "$ADDITIONAL_ZIP" ]]; then
  echo "Found additional data archive: $ADDITIONAL_ZIP"
  TMP_EXTRA="$(mktemp -d)"
  unzip -o "$ADDITIONAL_ZIP" -d "$TMP_EXTRA"

  for folder in depth_maps semantic_masks; do
    if [[ -d "$TMP_EXTRA/$folder" ]]; then
      echo "Moving $folder to session directory..."
      mkdir -p "$SESSION_DIR"
      mv -f "$TMP_EXTRA/$folder" "$SESSION_DIR/"
    else
      echo "Warning: '$folder' not found inside additional archive."
    fi
  done

  rm -rf "$TMP_EXTRA"
  rm -f "$ADDITIONAL_ZIP"
else
  echo "No additional data archive found. Skipping extra depth/mask import."
fi

echo "All downloads completed."
