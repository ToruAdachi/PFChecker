# Refactored Asset App (modular)

This is a modularized refactor of the original single-file Streamlit app.

## Run locally

```bash
streamlit run app.py
```

## External files directory

By default, external files are resolved **one directory above the app folder**:

- `data_j.xls` (JPX master for Kanji search)
- `portfolios.json` (portfolio presets)
- `instruments.csv` (fixed benchmark universe)
- cache files such as `jpx_listings_cache.csv`

On Raspberry Pi deployments where the app is placed at:

- `/home/pi/refactored_app_current/`

...external files should be placed at:

- `/home/pi/`

You can override the external data directory with:

```bash
export ASSET_APP_DATA_DIR=/path/to/data
```

## Raspberry Pi deployment

See `deploy/INSTALL_RPI.md`.
