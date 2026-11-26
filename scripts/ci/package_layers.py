#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 rokuz
# SPDX-License-Identifier: Apache-2.0
#
"""Pack the Graph and Tensor layers of a build directory into a release archive.

The archive holds the layer libraries and their manifests side by side. Each
manifest points at the library next to it, so the archive can be extracted
anywhere and added to VK_ADD_LAYER_PATH.
"""
import hashlib
import json
import pathlib
import sys
import zipfile

LAYERS = (("graph", "VkLayer_Graph"), ("tensor", "VkLayer_Tensor"))


def library_of(directory, stem):
    for name in (f"lib{stem}.so", f"lib{stem}.dylib", f"{stem}.dll"):
        if (directory / name).is_file():
            return directory / name
    sys.exit(f"{directory} has no {stem} library")


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: package_layers.py <build dir> <archive>")
    build = pathlib.Path(sys.argv[1])
    archive = pathlib.Path(sys.argv[2])
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for folder, stem in LAYERS:
            directory = build / folder
            library = library_of(directory, stem)
            zipped.write(library, library.name)
            manifest_path = directory / f"{stem}.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                separator = "\\" if library.suffix == ".dll" else "/"
                manifest["layer"]["library_path"] = f".{separator}{library.name}"
                zipped.writestr(f"{stem}.json", json.dumps(manifest, indent=4) + "\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"{digest}  {archive.name}")


if __name__ == "__main__":
    main()
