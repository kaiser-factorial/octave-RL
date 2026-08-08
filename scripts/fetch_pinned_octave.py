"""Unpack the pinned Octave image into a rootfs, without a Docker daemon.

Prime pods and CI runners are themselves containers, so ``docker run`` is
usually unavailable there. This pulls the ``gnuoctave/octave`` image's layers
straight from the registry over HTTPS and extracts them in order, producing a
directory tree that ``executors.py`` can ``chroot`` into.

The point is fidelity. ``apt install octave`` gives whatever the distro ships
-- 8.4.0 on Ubuntu 24.04 -- while the reference pool was validated against
10.2.0. Scoring on a different interpreter is a silent correctness risk on the
tolerance and orientation edge cases these tasks deliberately include.

Usage:
    uv run python scripts/fetch_pinned_octave.py --dest /opt/octave-rootfs
    export OCTAVE_RL_OCTAVE_ROOTFS=/opt/octave-rootfs

Requires root (or a user namespace) to chroot into the result, and about 4.5 GB
of disk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REGISTRY = "https://registry-1.docker.io"
AUTH = "https://auth.docker.io/token"
REPOSITORY = "gnuoctave/octave"
# Matches harness.OCTAVE_IMAGE. Keep the two in step.
DEFAULT_TAG = "10.2.0"
ARCH = "amd64"

MANIFEST_ACCEPT = (
    "application/vnd.oci.image.index.v1+json,"
    "application/vnd.docker.distribution.manifest.list.v2+json,"
    "application/vnd.oci.image.manifest.v1+json,"
    "application/vnd.docker.distribution.manifest.v2+json"
)


def _get(url: str, token: str | None = None, accept: str | None = None) -> bytes:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if accept:
        request.add_header("Accept", accept)
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def pull_token() -> str:
    url = f"{AUTH}?service=registry.docker.io&scope=repository:{REPOSITORY}:pull"
    return json.loads(_get(url))["token"]


def resolve_manifest(token: str, tag: str) -> dict:
    raw = json.loads(
        _get(f"{REGISTRY}/v2/{REPOSITORY}/manifests/{tag}", token, MANIFEST_ACCEPT)
    )
    if "manifests" not in raw:
        return raw
    digest = next(
        entry["digest"]
        for entry in raw["manifests"]
        if entry.get("platform", {}).get("architecture") == ARCH
        and entry.get("platform", {}).get("os") == "linux"
    )
    return json.loads(
        _get(f"{REGISTRY}/v2/{REPOSITORY}/manifests/{digest}", token, MANIFEST_ACCEPT)
    )


def fetch_rootfs(dest: Path, tag: str, cache: Path) -> None:
    token = pull_token()
    manifest = resolve_manifest(token, tag)
    layers = manifest["layers"]
    total_mb = sum(layer["size"] for layer in layers) / 1e6
    print(f"{REPOSITORY}:{tag} -- {len(layers)} layers, {total_mb:.0f} MB")

    dest.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for index, layer in enumerate(layers, 1):
        digest = layer["digest"]
        blob = cache / f"{digest.replace(':', '_')}.tar.gz"
        if blob.exists() and blob.stat().st_size == layer["size"]:
            print(f"  layer {index}/{len(layers)} cached")
        else:
            print(f"  layer {index}/{len(layers)} downloading "
                  f"({layer['size'] / 1e6:.0f} MB)", flush=True)
            blob.write_bytes(
                _get(f"{REGISTRY}/v2/{REPOSITORY}/blobs/{digest}", token)
            )
        # Layers are applied in order; later layers legitimately overwrite
        # earlier ones, which is why this is not extracted in parallel.
        _extract(blob, dest)
    print(f"rootfs ready at {dest}")


def _extract(blob: Path, dest: Path) -> None:
    """Extract one layer, preferring GNU tar over Python's tarfile.

    A distro image contains absolute symlinks -- ``etc/alternatives/awk`` points
    at ``/usr/bin/mawk`` -- and Python's ``filter="tar"`` rejects those as
    escaping the destination. Relaxing that to ``fully_trusted`` would accept a
    downloaded archive's word on where its members may land, so shell out to
    ``tar`` instead, which reproduces what a container runtime does.
    """
    if shutil.which("tar"):
        result = subprocess.run(
            ["tar", "-xzf", str(blob), "-C", str(dest)],
            capture_output=True,
            text=True,
            check=False,
        )
        # Whiteouts and unowned metadata produce warnings on a non-root
        # extraction; only a hard failure with no output written is fatal.
        if result.returncode != 0 and "Cannot" in result.stderr:
            raise RuntimeError(f"tar failed on {blob.name}: {result.stderr[:300]}")
        return
    with tarfile.open(blob) as archive:
        archive.extractall(dest, filter="tar")


def verify(dest: Path) -> int:
    for candidate in ("usr/local/bin/octave-cli", "usr/bin/octave-cli"):
        if (dest / candidate).exists():
            interpreter = f"/{candidate}"
            break
    else:
        print("FAIL: no octave-cli found in the extracted rootfs", file=sys.stderr)
        return 1
    probe = subprocess.run(
        ["chroot", str(dest), interpreter, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        print(
            "extracted, but could not chroot to verify "
            f"(needs root or a user namespace): {probe.stderr.strip()[:200]}"
        )
        return 0
    print(probe.stdout.splitlines()[0] if probe.stdout else "(no version output)")
    print(f"\nexport OCTAVE_RL_OCTAVE_ROOTFS={dest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("/opt/octave-rootfs"))
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/octave-image-cache"))
    args = parser.parse_args()
    fetch_rootfs(args.dest, args.tag, args.cache)
    return verify(args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
