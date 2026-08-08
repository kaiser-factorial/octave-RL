#!/usr/bin/env bash
# Bring a fresh Prime pod to the point where it can score Octave locally.
#
# Written to run unattended and be safe to re-run: each stage checks for its own
# output first, so a partial failure can be resumed without paying to redo the
# 1.6 GB image pull or the 8 GB model download.
#
# Deliberately does NOT touch Prime Sandboxes. Candidate scoring runs against
# the pinned Octave 10.2.0 rootfs fetched here, so the run has no dependency on
# the Sandbox scheduler.
#
# Usage:  bash scripts/pod_bootstrap.sh 2>&1 | tee /workspace/bootstrap.log

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
OCTAVE_RL="${OCTAVE_RL:-$WORKSPACE/octave_RL}"
PRIME_RL="${PRIME_RL:-$WORKSPACE/prime-rl}"
PRIME_RL_PIN="44539229436a23e624b0f39826014a4e58a703be"
ROOTFS="${ROOTFS:-/opt/octave-rootfs}"
BASE_MODEL="${BASE_MODEL:-$WORKSPACE/qwen3.5-4b-base}"
MERGED="${MERGED:-$WORKSPACE/step20-merged}"
ADAPTER="${ADAPTER:-$WORKSPACE/step20-adapter}"

step() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$1"; }

step "host"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || echo "no nvidia-smi"
df -h "$WORKSPACE" | tail -1
python3 --version || true

step "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "prime-rl @ $PRIME_RL_PIN"
# prime-rl pins its submodules to git@github.com: URLs, which need an SSH key
# the pod does not have. Rewrite to HTTPS so `submodule update` works on a
# fresh pod instead of failing four times and aborting.
git config --global url."https://github.com/".insteadOf "git@github.com:"
if [ ! -d "$PRIME_RL/.git" ]; then
  git clone --filter=blob:none https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_RL"
fi
cd "$PRIME_RL"
git checkout -q "$PRIME_RL_PIN"
git submodule update --init --recursive -q
echo "prime-rl at $(git rev-parse --short HEAD)"

step "prime-rl deps"
uv sync

step "octave_rl environment"
uv pip install -e "$OCTAVE_RL/environments/octave_rl"

step "pinned Octave 10.2.0 rootfs"
# The reference pool was validated on 10.2.0. Distro packages ship 8.4.0, and
# these tasks include tolerance and orientation edge cases where that could
# differ, so fetch the exact image rather than apt-installing a lookalike.
if [ ! -x "$ROOTFS/usr/local/bin/octave-cli" ]; then
  uv run python "$OCTAVE_RL/scripts/fetch_pinned_octave.py" --dest "$ROOTFS"
else
  echo "rootfs already present at $ROOTFS"
fi
export OCTAVE_RL_OCTAVE_ROOTFS="$ROOTFS"
chroot "$ROOTFS" /usr/local/bin/octave-cli --version | head -1

step "local runtime smoke (no Prime Sandbox)"
cd "$OCTAVE_RL"
OCTAVE_RL_OCTAVE_ROOTFS="$ROOTFS" uv run --project "$PRIME_RL" \
  python scripts/validate_local_runtime.py --num-tasks 5 \
  --report "$WORKSPACE/bootstrap-runtime-check.json"

step "base model"
if [ ! -f "$BASE_MODEL/config.json" ]; then
  uv run --project "$PRIME_RL" hf download Qwen/Qwen3.5-4B --local-dir "$BASE_MODEL"
else
  echo "base model already present at $BASE_MODEL"
fi
du -sh "$BASE_MODEL"

step "merge step-20 adapter"
if [ ! -f "$MERGED/config.json" ]; then
  uv run --project "$PRIME_RL" python "$OCTAVE_RL/scripts/merge_lora_checkpoint.py" \
    --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output "$MERGED"
else
  echo "merged checkpoint already present at $MERGED"
fi
du -sh "$MERGED"

step "done"
echo "export OCTAVE_RL_OCTAVE_ROOTFS=$ROOTFS"
echo "base   : $BASE_MODEL"
echo "step20 : $MERGED"
