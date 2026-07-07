#!/usr/bin/env bash
# healthcheck_collectors.sh — alerte (log) si un collecteur GBFS n'ecrit plus.
# Verifie a la fois la fraicheur des parquet et la presence des process.
# OK/ALERT ecrit dans logs/health.log ; les ALERT dupliquees dans logs/health_ALERT.log.
set -u
ROOT=/root/Recherche/bikeshare-data-explorer
LOG="$ROOT/logs/health.log"
ALERT="$ROOT/logs/health_ALERT.log"
MAX_AGE_H=2
now=$(date '+%Y-%m-%d %H:%M:%S')
status=OK
msg=""

check_store () {
  local name="$1" dir="$2" newest age_s age_h
  newest=$(find "$dir" -name '*.parquet' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  if [ -z "$newest" ]; then status=ALERT; msg="$msg [$name:aucun-parquet]"; return; fi
  age_s=$(( $(date +%s) - ${newest%.*} )); age_h=$(( age_s / 3600 ))
  if [ "$age_s" -gt $(( MAX_AGE_H * 3600 )) ]; then
    status=ALERT; msg="$msg [$name:rien-depuis-${age_h}h]"
  else
    msg="$msg [$name:ok-${age_h}h]"
  fi
}
proc_check () {
  local name="$1" pat="$2"
  pgrep -f "$pat" >/dev/null || { status=ALERT; msg="$msg [$name:process-MORT]"; }
}

check_store vehicles "$ROOT/data/vehicle_snapshots"
check_store status   "$ROOT/data/status_snapshots"
check_store alertes  "$ROOT/data/transit_alerts"
proc_check  vehicles '[c]ollect_vehicles\.py'
proc_check  status   '[c]ollect_status\.py'

# Alertes transport : une écriture-KO = perte de données imminente
# (leçon des 3-5 juillet 2026 : 12 passages TaM perdus en silence).
# 16 lignes de log = ~4 h de passages cron 15 min.
ko=$(tail -n 16 "$ROOT/logs/transit_alerts.log" 2>/dev/null | grep -c 'écriture-KO' || true)
if [ "${ko:-0}" -gt 0 ]; then status=ALERT; msg="$msg [alertes:ecriture-KO-x$ko]"; fi
rescues=$(find "$ROOT/data/transit_alerts" -name 'RESCUE_*.json' 2>/dev/null | wc -l)
if [ "${rescues:-0}" -gt 0 ]; then status=ALERT; msg="$msg [alertes:${rescues}-fichier(s)-RESCUE-a-reprendre]"; fi

line="$now $status$msg"
echo "$line" >> "$LOG"
if [ "$status" = ALERT ]; then echo "$line" >> "$ALERT"; echo "$line" >&2; fi
