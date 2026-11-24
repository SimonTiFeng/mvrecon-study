#!/usr/bin/env bash
# Resumable, verified upload: one stream at a time, per-segment size+hash check, retries.
# Usage: upload_verified.sh <local_file> <remote_dir> <remote_name> [segment_mb]
set -uo pipefail

LOCAL=$1; RDIR=$2; RNAME=$3; SEGMB=${4:-64}
SSH="ssh -o ServerAliveInterval=15 -o ConnectTimeout=20 -p 2222 aiserver@83.168.91.82"
SCP="scp -o ServerAliveInterval=15 -o ConnectTimeout=20 -P 2222"
SEG=$(( SEGMB * 1024 * 1024 ))
PY=$(cd "$(dirname "$0")/.." && pwd)/.venv/Scripts/python.exe
TOTAL=$(stat -c %s "$LOCAL")
NSEG=$(( (TOTAL + SEG - 1) / SEG ))
TMP=/tmp/upseg.$$
mkdir -p "$TMP"
echo "file=$LOCAL size=$TOTAL segments=$NSEG segbytes=$SEG"

bash -c "$SSH 'mkdir -p $RDIR/.parts'"

ok=0
for ((i=0; i<NSEG; i++)); do
  OFF=$(( i * SEG ))
  LEN=$(( TOTAL - OFF )); [ $LEN -gt $SEG ] && LEN=$SEG
  PART=$(printf "%s.seg%04d" "$RNAME" "$i")
  dd if="$LOCAL" of="$TMP/$PART" bs=1048576 skip=$((OFF/1048576)) \
     count=$(( (LEN + 1048575) / 1048576 )) iflag=skip_bytes,count_bytes status=none
  LHASH=$(sha256sum "$TMP/$PART" | awk '{print $1}')
  LSIZE=$(stat -c %s "$TMP/$PART")
  for attempt in 1 2 3 4 5; do
    RSIZE=$($SSH "stat -c %s $RDIR/.parts/$PART 2>/dev/null || echo 0")
    if [ "$RSIZE" = "$LSIZE" ]; then
      RHASH=$($SSH "sha256sum $RDIR/.parts/$PART 2>/dev/null | awk '{print \$1}'")
      if [ "$RHASH" = "$LHASH" ]; then ok=$((ok+1)); break; fi
    fi
    $SCP -q "$TMP/$PART" "aiserver@83.168.91.82:$RDIR/.parts/$PART" 2>/dev/null
  done
  echo "seg $i/$((NSEG-1)) size=$LSIZE attempts=$attempt"
done
rm -rf "$TMP"

echo "verified segments: $ok / $NSEG"
[ "$ok" -eq "$NSEG" ] || { echo "SEGMENTS_INCOMPLETE"; exit 1; }

$SSH "cd $RDIR/.parts && cat ${RNAME}.seg* > ../$RNAME && cd .. && rm -rf .parts && sha256sum $RNAME"
echo "REMOTE_SIZE=$($SSH "stat -c %s $RDIR/$RNAME")"
echo "LOCAL_SIZE=$TOTAL"
echo "UPLOAD_OK"
