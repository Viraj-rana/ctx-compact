"""Download LoCoMo (snap-research/locomo, CC BY-NC 4.0) into data/locomo/."""

from __future__ import annotations

import urllib.request

from ctxc.eval.locomo import URL, default_path


def main() -> None:
    path = default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, path)
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
