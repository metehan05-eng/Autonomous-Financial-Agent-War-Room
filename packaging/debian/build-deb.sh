#!/usr/bin/env bash
#
# Build a .deb for Debian/Ubuntu from the current checkout.
#
#   packaging/debian/build-deb.sh            # uses the version in pyproject
#   VERSION=0.2.0 packaging/debian/build-deb.sh
#
# The package stages an isolated virtualenv under /usr/lib/financial-agent-war-room
# and exposes the `financial-warroom` command in /usr/bin. Building a .deb is only
# the first step; see packaging/README.md for publishing a signed APT repository.
set -euo pipefail

PKG=financial-agent-war-room
ARCH=all
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="${VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -n1)}"

if [[ -z "${VERSION}" ]]; then
    echo "error: could not determine version (set VERSION=...)" >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo ">> building wheel for ${PKG} ${VERSION}"
python3 -m build --wheel --outdir "$STAGE/wheels" "$ROOT" >/dev/null
WHEEL="$(find "$STAGE/wheels" -name '*.whl' -print -quit)"

DEB_ROOT="$STAGE/${PKG}_${VERSION}_${ARCH}"
mkdir -p "$DEB_ROOT/DEBIAN" \
         "$DEB_ROOT/usr/lib/${PKG}" \
         "$DEB_ROOT/usr/bin"

sed "s/@VERSION@/${VERSION}/" "$ROOT/packaging/debian/control" > "$DEB_ROOT/DEBIAN/control"

echo ">> creating virtualenv"
python3 -m venv "$DEB_ROOT/usr/lib/${PKG}/venv"
"$DEB_ROOT/usr/lib/${PKG}/venv/bin/pip" install --no-cache-dir --upgrade pip >/dev/null
"$DEB_ROOT/usr/lib/${PKG}/venv/bin/pip" install --no-cache-dir "$WHEEL" >/dev/null

ln -s "/usr/lib/${PKG}/venv/bin/financial-warroom" "$DEB_ROOT/usr/bin/financial-warroom"

mkdir -p "$ROOT/dist"
OUT="$ROOT/dist/${PKG}_${VERSION}_${ARCH}.deb"
echo ">> packaging ${OUT}"
dpkg-deb --build --root-owner-group "$DEB_ROOT" "$OUT" >/dev/null
echo "done: ${OUT}"
