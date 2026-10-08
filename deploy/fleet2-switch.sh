#!/bin/bash
# fleet2: switch this worker between the v1 agent (polymarket-fleet) and the v2 agent.
#   sudo fleet2 use v2     stop and disable v1, enable and start v2
#   sudo fleet2 use v1     stop and disable v2, enable and start v1
#   fleet2 status          which agent is running
# Neither agent is uninstalled: both stay on disk with their identities.
# Before `use v2`, disable this box in the v1 dashboard (and never switch a box that is
# in v1's trade role); before `use v1`, let its v2 job finish or cancel it, so no job is
# cut short (a cut-short job shows as failed and can be run again on another worker).
# systemd's Conflicts= also stops the other agent if both are
# ever started.
set -euo pipefail
V1=fleet-worker.service
V2=fleet2-worker.service
state() { systemctl is-active "$1" 2>/dev/null || true; }
case "${1:-status}" in
  status)
    echo "v1 ($V1): $(state $V1)"
    echo "v2 ($V2): $(state $V2)"
    ;;
  use)
    [ "$(id -u)" = 0 ] || { echo "run as root: sudo fleet2 use ${2:-v2}" >&2; exit 1; }
    case "${2:-}" in
      v2)
        systemctl list-unit-files "$V2" | grep -q "$V2" || { echo "fleet-v2 is not installed" >&2; exit 1; }
        systemctl disable --now "$V1" >/dev/null 2>&1 || true
        systemctl enable --now "$V2"
        ;;
      v1)
        systemctl list-unit-files "$V1" | grep -q "$V1" || { echo "the v1 worker is not installed on this box" >&2; exit 1; }
        systemctl disable --now "$V2" >/dev/null 2>&1 || true
        systemctl enable --now "$V1"
        ;;
      *) echo "usage: sudo fleet2 use v1|v2" >&2; exit 2 ;;
    esac
    sleep 2
    echo "v1 ($V1): $(state $V1)"
    echo "v2 ($V2): $(state $V2)"
    ;;
  *) echo "usage: fleet2 status | sudo fleet2 use v1|v2" >&2; exit 2 ;;
esac
