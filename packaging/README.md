# Native packages (apt / Homebrew / winget)

These are **starting-point skeletons** for distributing the app through native
package managers. The recommended, fully working path is PyPI + pipx
(`pipx install financial-agent-war-room`); the files here show how to wrap the
same wheel for each platform.

| Channel | Directory | Status |
|---------|-----------|--------|
| Debian/Ubuntu (`apt`) | `debian/` | build script provided, needs a signed repo to `apt install` |
| macOS (`brew`) | `homebrew/` | formula skeleton, needs the PyPI release + `brew update-python-resources` |
| Windows (`winget`) | `winget/` | manifest skeleton, expects a PyInstaller `zip`/`portable` build |

## Why `sudo apt install` needs its own repo

`apt` only knows about repositories you have added. To make
`sudo apt install financial-agent-war-room` work for everyone you must:

1. build the `.deb` (`debian/build-deb.sh`),
2. host it in an APT repository with `Packages`/`Release` indices,
3. sign the repository with a GPG key,
4. have users add the repo + key once
   (`/etc/apt/sources.list.d/…` and `/etc/apt/keyrings/…`).

Only then does a plain `apt install` resolve the package. This only covers
Debian/Ubuntu — macOS and Windows use Homebrew and winget respectively; there
is no single command that covers all three operating systems.

## Building the Debian package

```bash
packaging/debian/build-deb.sh          # writes dist/financial-agent-war-room_*.deb
sudo dpkg -i dist/financial-agent-war-room_*.deb
```

## Notes

- The `.deb` stages an isolated virtualenv under `/usr/lib/financial-agent-war-room`.
  For a production package prefer installing into the system Python or creating
  the virtualenv from `postinst` so it is not relocated.
- The Homebrew formula currently expects a PyPI release tarball; fill in `url`
  and `sha256`, then run `brew update-python-resources` to add the dependency
  `resource` blocks.
- The winget manifest expects a self-contained Windows build (e.g. produced by
  [PyInstaller](https://pyinstaller.org/)) zipped and attached to a GitHub
  release; update the `InstallerUrl`/`InstallerSha256` fields accordingly.
