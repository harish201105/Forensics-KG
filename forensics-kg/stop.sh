#!/usr/bin/env bash
#
# Stop the Forensics-KG stack (frontend, backend, and the Neo4j Desktop DBMS).
# Your data is on disk and is NOT touched.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ok() { printf "  \033[32m✓\033[0m %s\n" "$1"; }

# Frontend (Next.js) + Backend (uvicorn)
pkill -f "uvicorn app.main:app" 2>/dev/null && ok "Backend stopped" || ok "Backend not running"
pkill -f "next dev"            2>/dev/null && ok "Frontend stopped" || ok "Frontend not running"

# Neo4j Desktop DBMS
DBMS_DIR="$(ls -d "$HOME/Library/Application Support/neo4j-desktop/Application/Data/dbmss/dbms-"* 2>/dev/null | head -1 || true)"
if [ -n "$DBMS_DIR" ]; then
  JAVA_HOME="$(ls -d "$HOME/Library/Application Support/neo4j-desktop/Application/Cache/runtime/zulu"* 2>/dev/null | sort | tail -1 || true)"
  export JAVA_HOME
  "$DBMS_DIR/bin/neo4j" stop >/dev/null 2>&1 && ok "Neo4j DBMS stopped" || ok "Neo4j DBMS not running (or managed by Desktop GUI)"
fi
echo "Done."
