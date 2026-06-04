#!/usr/bin/env bash
#
# Forensics-KG one-command launcher.
#
# Brings up all three tiers and waits until each is healthy:
#   1. Neo4j   (Neo4j Desktop DBMS that holds the `neo4j` database with your data)
#   2. Backend (FastAPI / uvicorn on :8000)
#   3. Frontend(Next.js dev server on :3000)
#
# Anything already running is left alone, so re-running this is safe.
#
# NOTE on Neo4j: your graph lives in a Neo4j *Desktop*-managed DBMS. This script
# starts that DBMS directly via its bundled JDK. Do NOT also press "Start" in the
# Neo4j Desktop GUI afterwards — two starts fight over port 7687. Stop with stop.sh.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
BACKEND_LOG="/tmp/forensics-backend.log"
FRONTEND_LOG="/tmp/forensics-frontend.log"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
info() { printf "  • %s\n" "$1"; }
die()  { printf "  \033[31m✗ %s\033[0m\n" "$1"; exit 1; }

port_up() { nc -z localhost "$1" >/dev/null 2>&1; }

# ── 1. Neo4j (Neo4j Desktop DBMS) ──────────────────────────────────────────────
bold "[1/3] Neo4j"
if port_up 7687; then
  ok "Neo4j already listening on :7687"
else
  DBMS_DIR="$(ls -d "$HOME/Library/Application Support/neo4j-desktop/Application/Data/dbmss/dbms-"* 2>/dev/null | head -1 || true)"
  [ -n "$DBMS_DIR" ] || die "No Neo4j Desktop DBMS found. Start it manually from the Neo4j Desktop app."
  JAVA_HOME="$(ls -d "$HOME/Library/Application Support/neo4j-desktop/Application/Cache/runtime/zulu"* 2>/dev/null | sort | tail -1 || true)"
  [ -n "$JAVA_HOME" ] || die "No bundled JDK found under Neo4j Desktop runtime."
  export JAVA_HOME
  info "Starting DBMS: $(basename "$DBMS_DIR")"
  "$DBMS_DIR/bin/neo4j" start >/dev/null 2>&1 || die "Failed to launch Neo4j DBMS."
  for _ in $(seq 1 30); do port_up 7687 && break; sleep 2; done
  port_up 7687 && ok "Neo4j up on :7687" || die "Neo4j did not come up within 60s (check $DBMS_DIR/logs/neo4j.log)."
fi

# ── 2. Backend (FastAPI) ───────────────────────────────────────────────────────
bold "[2/3] Backend"
if port_up 8000; then
  ok "Backend already listening on :8000"
else
  [ -x "$BACKEND/.venv/bin/uvicorn" ] || {
    info "Creating venv + installing deps (first run)…"
    python3 -m venv "$BACKEND/.venv"
    "$BACKEND/.venv/bin/pip" install -q --upgrade pip
    "$BACKEND/.venv/bin/pip" install -q -r "$BACKEND/requirements.txt"
  }
  ( cd "$BACKEND" && nohup .venv/bin/uvicorn app.main:app --port 8000 >"$BACKEND_LOG" 2>&1 & )
  for _ in $(seq 1 20); do curl -sf http://localhost:8000/api/health >/dev/null 2>&1 && break; sleep 2; done
  H="$(curl -s http://localhost:8000/api/health 2>/dev/null || true)"
  case "$H" in
    *'"neo4j":"connected"'*) ok "Backend up on :8000 (Neo4j connected)";;
    *'"status":"ok"'*)       ok "Backend up on :8000 (warning: Neo4j NOT connected)";;
    *) die "Backend failed to start. See $BACKEND_LOG";;
  esac
fi

# ── 3. Frontend (Next.js) ──────────────────────────────────────────────────────
bold "[3/3] Frontend"
if port_up 3000; then
  ok "Frontend already listening on :3000"
else
  [ -d "$FRONTEND/node_modules" ] || {
    info "Installing npm deps (first run)…"
    ( cd "$FRONTEND" && npm install --silent )
  }
  ( cd "$FRONTEND" && nohup npm run dev >"$FRONTEND_LOG" 2>&1 & )
  for _ in $(seq 1 30); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:3000 2>/dev/null)" = "200" ] && break; sleep 2; done
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:3000 2>/dev/null)" = "200" ] && ok "Frontend up on :3000" || die "Frontend failed to start. See $FRONTEND_LOG"
fi

echo
bold "Forensics-KG is running:"
NODES="$(curl -s http://localhost:8000/api/graph/stats 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("total_nodes","?"),"nodes /",d.get("total_relationships","?"),"relationships")' 2>/dev/null || echo "stats unavailable")"
info "Frontend : http://localhost:3000"
info "Backend  : http://localhost:8000  (docs: /docs)"
info "Graph    : $NODES"
info "Logs     : $BACKEND_LOG , $FRONTEND_LOG"
echo "  Stop everything with: ./stop.sh"
