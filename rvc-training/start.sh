#!/bin/sh
# Starts the worker, optionally with the newest handler.py from GitHub so a
# handler change needs only a push, not an image rebuild.
#   HANDLER_REPO  owner/repo to fetch from (unset: use the baked handler)
#   HANDLER_REF   branch or tag (default main)
#   GITHUB_TOKEN  only needed if the repo is private
# If the download fails or doesn't compile, the baked handler.py runs.
set -u
handler=/worker/handler.py
export HANDLER_SOURCE="baked into the image"

if [ -n "${HANDLER_REPO:-}" ]; then
  ref="${HANDLER_REF:-main}"
  api="https://api.github.com/repos/$HANDLER_REPO"
  auth=""
  [ -n "${GITHUB_TOKEN:-}" ] && auth="Authorization: Bearer $GITHUB_TOKEN"
  # A stalled GitHub can't hold the worker up: give up and use the baked handler.
  wait="--connect-timeout 5 --max-time 20"
  if curl -fsS $wait ${auth:+-H "$auth"} -H "Accept: application/vnd.github.raw" \
       "$api/contents/rvc-training/handler.py?ref=$ref" -o /tmp/handler.py \
     && python -m py_compile /tmp/handler.py; then
    handler=/tmp/handler.py
    sha=$(curl -fsS $wait ${auth:+-H "$auth"} -H "Accept: application/vnd.github.sha" "$api/commits/$ref" || echo unknown)
    export HANDLER_SOURCE="$HANDLER_REPO@$ref@$sha"
  else
    echo "couldn't fetch handler.py from $HANDLER_REPO@$ref; using the baked one" >&2
  fi
fi

echo "handler: $HANDLER_SOURCE"
exec python -u "$handler"
