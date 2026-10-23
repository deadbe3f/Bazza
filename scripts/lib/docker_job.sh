# Shared container entrypoint: durable logs/exit marker and graceful forwarding.
# Caller supplies Docker lifetime controls and passes output-dir, then argv.
bazza_job_entrypoint='
output=$1; shift
rm -f "$output/exit_code.txt"
job_started=$(date +%s)
"$@" > >(tee -a "$output/console.log") 2>&1 & job_pid=$!
trap '\''kill -TERM "$job_pid" 2>/dev/null || true'\'' TERM INT
while true; do
  wait "$job_pid"; result=$?
  if ! kill -0 "$job_pid" 2>/dev/null; then break; fi
done
printf "%s\n" "$result" > "$output/exit_code.txt"
printf "%s\t%s\t%s\n" "$job_started" "$(date +%s)" "$result" >> "$output/job-sessions.tsv"
exit "$result"
'
