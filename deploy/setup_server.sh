#!/bin/bash
# One-time setup of the arXiv Paper Assistant on an Ubuntu 24.04 server (x86_64 or ARM64), run as the admin user.
# Expects the data disk mounted at /data, the app code in /data/app (with its .env), and optionally a Qdrant
# snapshot in /data/qdrant/snapshots/. Safe to run again: every step checks what is already there.
#
# Afterwards two systemd services start at every boot:
#   qdrant.service      Qdrant on localhost:6333, data in /data/qdrant/storage
#   arxiv-app.service   the Chainlit app on port 8000 (waits for Qdrant)
set -euo pipefail

QDRANT_VERSION="1.20.0"
APP_DIR=/data/app
VENV=/data/venv
QDRANT_DIR=/data/qdrant
RUN_USER="${SUDO_USER:-$USER}"

case "$(uname -m)" in
  aarch64|arm64) QDRANT_ASSET="qdrant-aarch64-unknown-linux-musl.tar.gz" ;;
  x86_64)        QDRANT_ASSET="qdrant-x86_64-unknown-linux-musl.tar.gz" ;;
  *) echo "Unsupported CPU: $(uname -m)"; exit 1 ;;
esac

echo "== 1. system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates >/dev/null

echo "== 2. Qdrant ${QDRANT_VERSION} ($(uname -m))"
mkdir -p "$QDRANT_DIR/storage" "$QDRANT_DIR/snapshots"
if [ ! -x "$QDRANT_DIR/qdrant" ] || ! "$QDRANT_DIR/qdrant" --version 2>/dev/null | grep -q "$QDRANT_VERSION"; then
  curl -sSL "https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}/${QDRANT_ASSET}" \
    | tar -xz -C "$QDRANT_DIR"
fi
"$QDRANT_DIR/qdrant" --version

echo "== 3. Python 3.13 + packages (uv)"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$VENV/bin/python" ] || "$UV" venv -q --python 3.13 "$VENV"
# CPU-only PyTorch: the server has no GPU, and the CPU wheels are much smaller.
"$UV" pip install -q --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install -q --python "$VENV/bin/python" -r "$APP_DIR/requirements.txt"
"$VENV/bin/python" -c "import torch, chainlit, qdrant_client; print('torch', torch.__version__, '| chainlit', chainlit.__version__)"

echo "== 4. systemd services"
sudo tee /etc/systemd/system/qdrant.service >/dev/null <<EOF
[Unit]
Description=Qdrant vector database
After=network-online.target data.mount
RequiresMountsFor=/data

[Service]
User=$RUN_USER
WorkingDirectory=$QDRANT_DIR
Environment=QDRANT__STORAGE__STORAGE_PATH=$QDRANT_DIR/storage
Environment=QDRANT__STORAGE__SNAPSHOTS_PATH=$QDRANT_DIR/snapshots
Environment=QDRANT__SERVICE__HOST=127.0.0.1
ExecStart=$QDRANT_DIR/qdrant
Restart=on-failure
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/arxiv-app.service >/dev/null <<EOF
[Unit]
Description=arXiv Paper Assistant (Chainlit)
After=qdrant.service
Requires=qdrant.service
RequiresMountsFor=/data

[Service]
User=$RUN_USER
WorkingDirectory=$APP_DIR
ExecStartPre=/bin/bash -c 'for i in \$(seq 1 120); do curl -sf http://127.0.0.1:6333/collections >/dev/null && exit 0; sleep 2; done; exit 1'
ExecStart=$VENV/bin/python -m chainlit run threeofus_main.py -h --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable qdrant.service arxiv-app.service >/dev/null
sudo systemctl restart qdrant.service
echo "Setup done. Restore the snapshot (if not done yet), then: sudo systemctl restart arxiv-app"
