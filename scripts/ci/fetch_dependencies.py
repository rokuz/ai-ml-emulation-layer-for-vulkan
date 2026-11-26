#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 rokuz
# SPDX-License-Identifier: Apache-2.0
#
"""Clone the pinned source dependencies that the release binaries are built against."""
import pathlib
import subprocess
import sys

DEPENDENCIES = {
    "glslang": ("https://github.com/KhronosGroup/glslang", "168d452a4f460d24b588fed08477a81c44ee27a1"),
    "spirv-cross": ("https://github.com/KhronosGroup/SPIRV-Cross", "6c09849fe88c48eaed08413aa022aaa136a3a057"),
    "spirv-headers": ("https://github.com/KhronosGroup/SPIRV-Headers", "29981f65241605e08b0ede4cfeb999fe3b723c6a"),
    "spirv-tools": ("https://github.com/KhronosGroup/SPIRV-Tools", "9a49b0883b9b635689a85b5647dbfcb223268151"),
    "vulkan-headers": ("https://github.com/KhronosGroup/Vulkan-Headers", "e3b1eec08173d6b825cd3ac88c885a63b621504a"),
    "json": ("https://github.com/nlohmann/json", "55f93686c01528224f448c19128836e7df245f72"),
}


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: fetch_dependencies.py <destination>")
    destination = pathlib.Path(sys.argv[1]).resolve()
    for name, (url, commit) in DEPENDENCIES.items():
        checkout = destination / name
        checkout.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=checkout)
        git("fetch", "-q", "--depth", "1", url, commit, cwd=checkout)
        git("checkout", "-q", "FETCH_HEAD", cwd=checkout)
        print(f"{name} {commit}")


if __name__ == "__main__":
    main()
