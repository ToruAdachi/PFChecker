# Codex 用プロジェクトルール（PFChecker）

## 目的 / 優先順位
- 変更は最小リスクで実装する。
- 変更が明示されていない限り、既存挙動を維持する。
- 差分は小さく、レビューしやすく保つ。
- Raspberry Pi 実機での挙動を最優先とし、ローカル PC との差異に注意する。

## 言語 / 実行環境
- Python 3.x
- Streamlit
- 依存管理（poetry/pipenv）は存在する場合のみ前提とする。
- 実行環境は Raspberry Pi（systemd 管理）を前提とする。

## 実行方法（確定）
### Raspberry Pi（本番相当）
- アプリ本体は以下のディレクトリに配置されている前提：
  `/home/pi/refactored_app_current`
- systemd サービス定義：
  `/etc/systemd/system/streamlit-app.service`
- 正式な起動・停止方法：
```bash
sudo systemctl stop streamlit-app
sudo systemctl start streamlit-app
sudo systemctl restart streamlit-app
```
- 手動起動（デバッグ用途のみ）：
```bash
cd /home/pi/refactored_app_current
streamlit run app.py
```
Codex / ChatGPT は run コマンドを「推測」してはならない。必ず上記の前提を守ること。

## 更新 / デプロイ手順（Raspberry Pi）
ZIP または GitHub から更新する場合の基本手順：
```bash
sudo systemctl stop streamlit-app

# 例：ディレクトリを丸ごと差し替える場合
rm -rf /home/pi/refactored_app_current
mv refactored_app /home/pi/refactored_app_current

sudo systemctl start streamlit-app
```
systemctl 停止中にのみファイルを更新すること。稼働中にファイルを書き換えない。

## テスト（自動テストなし）
自動テストフレームワークは存在しない。

pytest / unittest / CI は導入しない（明示的に依頼された場合を除く）。

## 手動 / 画面確認テスト（確定）
以下が正式なテスト手法である：
- Streamlit UI をブラウザで確認（PC / スマホ両方）
- 確認項目例：
  - 起動直後にエラーが出ないこと
  - チャートが表示されること
  - 銘柄追加・削除で UI が壊れないこと
  - カーソル移動に応じて表が更新されること
  - ラズパイ再起動後も自動起動すること
- journalctl によるログ確認：
```bash
journalctl -u streamlit-app -n 100 --no-pager
```

## コードスタイル / リファクタリング
既存スタイルに従う。

無関係な整形やリファクタリングは行わない。

明示的な依頼がない限り、大きな構造変更は避ける。

JavaScript / Streamlit Component は特に壊れやすいため、慎重に扱う。

## ファイル / 安全性
必要なファイルのみ変更する。

明示的な指示がない限り、ファイルを削除しない。

以下は変更しない（明示的に依頼された場合を除く）：
- `.gitignore`
- project structure
- dependencies

## Git 運用
自動でコミットしない。

明示的に依頼されない限り、履歴の書き換えや squash は行わない。

作業の最後に以下を要約する：
- 変更したファイル
- 変更の意図

## Codex / LLM 注意事項
Raspberry Pi + Streamlit + systemd という制約を常に意識すること。

「一見正しそうな改善」でも UX・実機挙動が悪化する変更は不可。

不明点があれば 実装前に必ず確認すること。
