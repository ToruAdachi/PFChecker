# Raspberry Pi (systemd + symlink) deployment

This project is designed to run via **systemd** using a **stable symlink** so you can update by switching the symlink.

## Directory convention

- App code (symlink target):
  - `/home/pi/refactored_app_current/`  (symlink)
- Release directories:
  - `/home/pi/releases/refactored_app_YYYYMMDD_HHMMSS/`
- External data files (one level above app dir):
  - `/home/pi/data_j.xls`
  - `/home/pi/portfolios.json`
  - other csv/json caches created by the app

Optional (recommended for stable U.S. stock suggestions):
  - `/home/pi/us_tickers.csv` (columns: symbol,name,exchange,asset_class)
    - This project includes a builder script that generates a **NYSE + NASDAQ** universe and appends crypto symbols
      (**BTC-USD**, **ETH-USD**). See "3) Build / update us_tickers.csv".
    - A small sample is bundled in this zip at: `refactored_app/data/us_tickers_sample.csv`

You can override the external data directory with an environment variable:

```bash
export ASSET_APP_DATA_DIR=/path/to/data
```

## 1) Install or update the systemd unit

Copy the unit template:

```bash
sudo cp -f /home/pi/refactored_app_current/deploy/streamlit-app.service /etc/systemd/system/streamlit-app.service
sudo systemctl daemon-reload
sudo systemctl enable streamlit-app
```

Start it:

```bash
sudo systemctl start streamlit-app
```

Check status/logs:

```bash
sudo systemctl status streamlit-app
journalctl -u streamlit-app -n 200 --no-pager
```

## 2) Deploy an update (symlink switch)

Assuming you have a new zip file at `/home/pi/refactored_app_v2_full.zip`:

```bash
sudo systemctl stop streamlit-app

ts=$(date +%Y%m%d_%H%M%S)
mkdir -p /home/pi/releases/refactored_app_${ts}
unzip -q /home/pi/refactored_app_v2_full.zip -d /home/pi/releases/refactored_app_${ts}

# If the zip contains a top-level folder (recommended), adjust the target path accordingly.
# This zip is packaged with a top-level folder "refactored_app".
ln -sfn /home/pi/releases/refactored_app_${ts}/refactored_app /home/pi/refactored_app_current

sudo systemctl start streamlit-app
```

Rollback (switch to previous release dir) is the same: point the symlink back and restart.

## 3) Build / update us_tickers.csv (NYSE + NASDAQ) and enable daily update

### One-off build (writes /home/pi/us_tickers.csv)

```bash
/home/pi/venv/bin/pip install -U requests
/home/pi/venv/bin/python /home/pi/refactored_app_current/deploy/build_us_ticker_dict.py
ls -l /home/pi/us_tickers.csv
```

### Enable daily update (systemd timer)

```bash
sudo cp -f /home/pi/refactored_app_current/deploy/us-tickers-update.service /etc/systemd/system/us-tickers-update.service
sudo cp -f /home/pi/refactored_app_current/deploy/us-tickers-update.timer /etc/systemd/system/us-tickers-update.timer
sudo systemctl daemon-reload
sudo systemctl enable --now us-tickers-update.timer
systemctl list-timers --all | grep us-tickers-update
```

Check logs:

```bash
journalctl -u us-tickers-update.service -n 200 --no-pager
```

