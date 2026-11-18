# Deploying to Azure (Azure for Students)

How the app went from a laptop (Ollama on a GPU, Qdrant in WSL) to an Azure VM that anyone with the shared
login can open in a browser. Every step below was run once on 2026-11-16 and works as written.

**What runs where**

| Part | Local (PC) | Azure VM |
|---|---|---|
| LLM | Ollama `qwen3:4b` on the GPU | Groq API, `qwen/qwen3.8-27b` (no GPU on the VM) |
| Qdrant | WSL, `~/qdrant/storage` | `/data/qdrant/storage`, listens on 127.0.0.1 only |
| Embeddings + reranker | GPU | CPU (PyTorch CPU build) |
| Chat history + accounts | `history.db` (SQLite) | `/data/app/history.db` (SQLite, on the data disk) |
| Login | each user signs up | one shared account, sign-up closed |

## 1. Settings in `.env`

The same code runs in both places; only `.env` differs. New settings on this branch:

| Setting | Values | Default | On the VM |
|---|---|---|---|
| `LLM_PROVIDER` | `ollama` / `groq` | `ollama` | `groq` |
| `GROQ_API_KEY` | key from console.groq.com | - | set |
| `GROQ_MODEL` | any model your key can use | `qwen/qwen3.8-27b` | default |
| `ALLOW_SIGNUP` | `1` / `0` | `1` | `1` |
| `SIGNUP_LIMIT` | most accounts the sign-up page may create (`adduser` accounts don't count) | empty = no limit | `12` |
| `ADMIN_USERS` | accounts that may open the read-only admin panel at `/admin` (comma-separated); make one with `python db.py adduser NAME` | empty = nobody | your admin account |
| `DB_BACKEND` | `sqlite` / `postgres` | `sqlite` | default (sqlite) |
| `DATABASE_URL` / `DATABASE_URL_UNPOOLED` | Postgres URL | - | not used |

The reranker cutoff is not an `.env` setting: each user sets it with the "Reranker cutoff" slider in Settings (0.6-0.95, default 0.7, `RERANK_CUTOFF` in threeofus_main.py).

`.env` is in `.gitignore`. Never commit it: it holds the Groq key and `CHAINLIT_AUTH_SECRET`.

## 2. Create the VM

Azure for Students only allows some regions and VM sizes, and gives some VM families a CPU quota of 0.
Check before choosing (read-only, no cost):

```bash
az policy assignment list --query "[].parameters.listOfAllowedLocations.value" -o tsv   # allowed regions
az vm list-usage --location japaneast -o table                                           # quota per family
```

On our subscription the only 16 GB size that was both unrestricted and had quota in Japan East was
**`Standard_B4ps_v2`** (4 vCPU ARM64, 16 GB, about $0.17/hour). `Standard_E2as_v7` was listed as
available but failed with `QuotaExceeded` (limit 0).

```bash
az provider register -n Microsoft.Compute --wait
az provider register -n Microsoft.Network --wait
az provider register -n Microsoft.Storage --wait
ssh-keygen -t ed25519 -f ~/.ssh/arxiv_azure -N "" -C arxiv-assistant-vm

az group create -n arxiv-assistant-rg -l japaneast
az vm create -g arxiv-assistant-rg -n arxiv-assistant-vm -l japaneast \
  --image Canonical:ubuntu-24_04-lts:server-arm64:latest --size Standard_B4ps_v2 \
  --admin-username azureuser --ssh-key-values ~/.ssh/arxiv_azure.pub \
  --public-ip-sku Standard --public-ip-address-allocation static \
  --os-disk-size-gb 32 --storage-sku StandardSSD_LRS \
  --data-disk-sizes-gb 64 --data-disk-delete-option Detach --nsg-rule SSH
az vm open-port -g arxiv-assistant-rg -n arxiv-assistant-vm --port 8000 --priority 1010
```

On Windows, `az login` may hang at the account picker with a school account; `az login --use-device-code` works.

## 3. Mount the data disk at `/data`

All app data lives on the separate data disk, so it survives a VM rebuild (detach it, attach it to a new VM).

```bash
sudo mkfs.ext4 -L arxivdata /dev/sdb          # only the first time: erases the disk
sudo mkdir -p /data
echo "UUID=$(sudo blkid -s UUID -o value /dev/sdb) /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo mount -a && sudo chown azureuser:azureuser /data
mkdir -p /data/qdrant/storage /data/qdrant/snapshots /data/app /data/models_cache
```

## 4. Copy the code and `.env`

From the repo folder on the PC (WSL), without git data, caches or the local chat history:

```bash
rsync -a --delete --exclude ".git/" --exclude "models_cache/" --exclude "history.db" --exclude ".files/" \
  --exclude "selections/" --exclude "__pycache__/" --exclude ".venv/" --exclude ".env" \
  -e "ssh -i ~/.ssh/arxiv_azure" ./ azureuser@<VM-IP>:/data/app/
scp -i ~/.ssh/arxiv_azure .env azureuser@<VM-IP>:/data/app/.env
```

Then on the VM: `chmod 600 /data/app/.env`, set `LLM_PROVIDER=groq`, `ALLOW_SIGNUP=1` and `SIGNUP_LIMIT=12`
(or `ALLOW_SIGNUP=0` to close sign-up), and make sure no
`DB_BACKEND=postgres` line is there. Files copied from Windows arrive with mode 777; fix them:

```bash
find /data/app -type d -exec chmod 755 {} + && find /data/app -type f -exec chmod 644 {} + && chmod 600 /data/app/.env
```

## 5. Install everything

```bash
bash /data/app/deploy/setup_server.sh
```

[setup_server.sh](setup_server.sh) installs Qdrant 1.20.0 (ARM64 or x86_64, picked automatically), Python 3.13
via uv, CPU-only PyTorch and `requirements.txt` into `/data/venv`, and two systemd services that start at boot:
`qdrant` and `arxiv-app` (Chainlit on port 8000; it waits until Qdrant answers).

## 6. Copy the Qdrant data (snapshot)

On the PC, with Qdrant running (about 1 minute, 7.3 GB file):

```bash
curl -X POST "localhost:6333/collections/sebun_bge768_hybrid_full/snapshots?wait=true"
rsync -a --partial -e "ssh -i ~/.ssh/arxiv_azure" ~/qdrant/snapshots/sebun_bge768_hybrid_full/ \
  azureuser@<VM-IP>:/data/qdrant/snapshots/
```

On the VM, check the checksum, then restore (about 2 minutes):

```bash
cd /data/qdrant/snapshots && sha256sum -c <(echo "$(cat *.checksum)  $(ls *.snapshot)")
curl -X PUT "127.0.0.1:6333/collections/sebun_bge768_hybrid_full/snapshots/recover?wait=true" \
  -H "Content-Type: application/json" -d "{\"location\": \"file:///data/qdrant/snapshots/$(ls *.snapshot)\"}"
```

The snapshot made on x86 restored fine on ARM64 (1,104,538 points). After a restore Qdrant shows status
`yellow` and rebuilds indexes for about 15 minutes at high CPU; searches are slow until it is `green`.
This happens once; later boots load the finished index from disk.

## 7. Start the app and create the shared account

```bash
sudo systemctl start arxiv-app
/data/venv/bin/python /data/app/db.py adduser <name>     # asks for the password twice (not shown)
```

Open `http://<VM-IP>:8000`. The first start downloads the embedding, BM25 and reranker models (about 420 MB)
into `/data/app/models_cache`.

## Day to day

```bash
az vm deallocate -g arxiv-assistant-rg -n arxiv-assistant-vm   # stop: compute billing stops, data is kept
az vm start      -g arxiv-assistant-rg -n arxiv-assistant-vm   # start: services come up by themselves
ssh -i ~/.ssh/arxiv_azure azureuser@<VM-IP> "journalctl -u arxiv-app -n 50"   # app log
az group delete -n arxiv-assistant-rg                          # delete everything when done
```

A stopped (deallocated) VM still pays for its disks and the static IP (about $0.40/day).

## Problems we hit

| Symptom | Cause | Fix |
|---|---|---|
| `QuotaExceeded` creating the VM | student quota 0 for that VM family | pick a size whose family has quota (`az vm list-usage`) |
| "Error fetching threads: Internal Server Error" after login | `greenlet` missing: SQLAlchemy's async mode needs it | `SQLAlchemy[asyncio]` in requirements.txt (fixed) |
| Groq 429 "Request too large" | uncapped `max_tokens` counted against 8,000 tokens/min | output capped in `groq_llm()` (fixed) |
| Groq 404 for `qwen/qwen3-32b` | model no longer offered | default is now `qwen/qwen3.8-27b` |
| Neon Postgres times out | campus network and Hugging Face Spaces block port 5432 | stayed on SQLite; Postgres is opt-in (`DB_BACKEND=postgres`) |
