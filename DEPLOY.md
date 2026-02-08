# Lab Dojo Deployment Guide

This guide covers deploying Lab Dojo to production using **Railway** (compute), **Supabase** (PostgreSQL, future), and **Cloudflare R2** (object storage, future).

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  React UI   │────▶│  Railway (Python) │────▶│  SQLite/WAL  │
│  (static)   │     │  FastAPI server   │     │  (local disk) │
└─────────────┘     └──────────────────┘     └──────────────┘
                           │                        │
                    ┌──────┴──────┐          ┌──────┴──────┐
                    │ 20 Science  │          │ Cloudflare  │
                    │ APIs (free) │          │ R2 (future) │
                    └─────────────┘          └─────────────┘
```

**Current (v0.1.2):** SQLite on local disk. Perfect for single-user local installs.

**Future (v0.2.0):** Supabase PostgreSQL for multi-user hosted mode, R2 for file exports.

---

## 1. Railway Deployment

### Prerequisites
- [Railway account](https://railway.app)
- GitHub repo connected to Railway

### Quick Deploy

1. **Connect your repo** to Railway from the dashboard
2. Railway auto-detects the `Dockerfile` and `railway.toml`
3. Set environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | Auto | Railway sets this automatically |
| `OPENAI_API_KEY` | Optional | For OpenAI backend |
| `ANTHROPIC_API_KEY` | Optional | For Anthropic backend |
| `VASTAI_API_KEY` | Optional | For Vast.ai serverless |

4. Deploy. The health check at `/status` confirms the server is running.

### Railway Configuration

The `railway.toml` configures:
- **Dockerfile build** with multi-stage (Node.js + Python)
- **Health check** on `/status` with 30s timeout
- **Auto-restart** on failure (max 3 retries)
- **Single replica** (scale up as needed)

### Scaling

Railway supports horizontal scaling. For Lab Dojo:
- **1 replica** handles ~100 concurrent users (API-bound, not CPU-bound)
- **Scale to 0** is supported — Railway sleeps idle services
- Add replicas for higher concurrency (each replica has its own SQLite)

> **Note:** SQLite is per-replica. For multi-replica deployments, migrate to Supabase PostgreSQL (Phase 3 roadmap).

---

## 2. Supabase (Future: v0.2.0)

When Lab Dojo moves to multi-user hosted mode:

### Setup
1. Create a Supabase project at [supabase.com](https://supabase.com)
2. Get the connection string from Settings → Database
3. Set `DATABASE_URL` in Railway environment variables

### Migration Plan
- Replace SQLite with `asyncpg` + Supabase PostgreSQL
- Schema is already designed for relational DB (see `KnowledgeBase._init_db`)
- Add Supabase Auth for user management
- Row-Level Security (RLS) for multi-tenant data isolation

### Cost Estimate
| Tier | Users | Cost |
|------|-------|------|
| Free | <500 | $0/mo |
| Pro | <10K | $25/mo |
| Team | <100K | $599/mo |

---

## 3. Cloudflare R2 (Future: v0.2.0)

For file exports (BibTeX, RIS, conversation logs):

### Setup
1. Create R2 bucket at [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Generate API token with R2 read/write permissions
3. Set environment variables:

```
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=labdojo-exports
R2_PUBLIC_URL=https://exports.labdojo.org
```

### Cost Estimate
| Usage | Cost |
|-------|------|
| Storage | $0.015/GB/mo |
| Class A ops (write) | $4.50/M |
| Class B ops (read) | $0.36/M |
| Egress | Free |

For 1M users generating ~10MB/mo each: ~$150/mo.

---

## 4. Local Development

```bash
# Clone the repo
git clone https://github.com/SooparAI/labdojo.git
cd labdojo

# Option A: Single-file (no build step)
pip install aiohttp fastapi uvicorn pydantic
python labdojo.py

# Option B: With React UI
cd labdojo-ui && pnpm install && pnpm build && cd ..
python labdojo.py

# Option C: Docker
docker build -t labdojo .
docker run -p 8080:8080 labdojo
```

---

## 5. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port (Railway sets automatically) |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `VASTAI_API_KEY` | - | Vast.ai serverless key |
| `NCBI_API_KEY` | - | NCBI API key (optional, higher rate limits) |
| `DATABASE_URL` | SQLite | PostgreSQL URL (future) |
| `R2_ACCOUNT_ID` | - | Cloudflare R2 account (future) |
| `R2_ACCESS_KEY_ID` | - | R2 access key (future) |
| `R2_SECRET_ACCESS_KEY` | - | R2 secret key (future) |
| `R2_BUCKET_NAME` | - | R2 bucket name (future) |

---

## 6. Monitoring

Lab Dojo exposes:
- `GET /status` — Server health, AI backend availability, API count
- `GET /apis` — Status of all 20 connected APIs
- Usage stats tracked in SQLite (`usage_stats` table)

For production monitoring, connect Railway's built-in metrics or add:
- **Sentry** for error tracking (add `sentry-sdk[fastapi]`)
- **Prometheus** metrics endpoint (future)
