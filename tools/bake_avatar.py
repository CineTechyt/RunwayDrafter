#!/usr/bin/env python3
"""Downloads the MakeHuman base mesh, morph targets and skeleton and bakes them into
`assets/avatar/body.bin`, the compact file the `avatar` crate loads.

MakeHuman's assets (mesh, targets, skeleton) are released under CC0 1.0; see
https://github.com/makehumancommunity/makehuman/blob/master/LICENSE.ASSETS.md
None of MakeHuman's (AGPL) program code is used.

Usage:  python3 tools/bake_avatar.py [cache_dir]

File layout (little endian), lengths in metres:
    magic      8 bytes  "C3BODY2\\0"
    u32        vertex count V (mesh vertices first, then joint markers)
    u32        mesh vertex count (the vertices triangles use)
    f32[3]*V   rest positions
    u32        triangle count T
    u32        skin triangle count (the first of the T; the rest are eyes)
    u32[3]*T   triangles
    u32        target count
    per target: u16 name length, name (utf-8), u32 entry count,
                entries of (u32 vertex, f32[3] offset)
    u32        bone count, parents before children
    per bone:  u16 name length, name, i32 parent (-1 = root),
               u32 n, u32*n head vertices, u32 n, u32*n tail vertices
               (a joint sits at the mean of its vertices, so it follows
               the morphs)
    per mesh vertex: u16[4] bones, f32[4] weights (sum 1)
"""

import concurrent.futures
import json
import os
import struct
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/"
DECIMETRE = 0.1  # MakeHuman models in decimetres

MUSCLE = ["minmuscle", "averagemuscle", "maxmuscle"]
WEIGHT = ["minweight", "averageweight", "maxweight"]
GENDER = ["female", "male"]
MEASURES = [
    "ankle-circ", "bust-circ", "calf-circ", "frontchest-dist", "hips-circ", "knee-circ",
    "lowerarm-length", "lowerleg-height", "napetowaist-dist", "neck-circ", "neck-height",
    "shoulder-dist", "thigh-circ", "underbust-circ", "upperarm-circ", "upperarm-length",
    "upperleg-height", "waist-circ", "waisttohip-dist", "wrist-circ",
]


def target_files():
    """(name in the baked file, path in the MakeHuman repository)"""
    files = []
    for gender in GENDER:
        for race in ["african", "asian", "caucasian"]:
            files.append((f"{race}-{gender}", f"targets/macrodetails/{race}-{gender}-young.target"))
        for muscle in MUSCLE:
            for weight in WEIGHT:
                combo = f"{gender}-young-{muscle}-{weight}"
                short = f"{gender}-{muscle}-{weight}"
                files.append((f"universal-{short}", f"targets/macrodetails/universal-{combo}.target"))
                for height in ["minheight", "maxheight"]:
                    files.append((f"{height}-{short}", f"targets/macrodetails/height/{combo}-{height}.target"))
                if gender == "female":
                    for cup in ["mincup", "maxcup"]:
                        files.append((f"{cup}-{short}", f"targets/breast/{combo}-{cup}-averagefirmness.target"))
    for measure in MEASURES:
        for direction in ["decr", "incr"]:
            files.append((f"{measure}-{direction}", f"targets/measure/measure-{measure}-{direction}.target"))
    return files


def fetch(cache, path):
    local = os.path.join(cache, path.replace("/", "__"))
    if not os.path.exists(local):
        with urllib.request.urlopen(RAW + path, timeout=120) as response:
            data = response.read()
        with open(local, "wb") as out:
            out.write(data)
    return local


def read_base(path):
    """Returns (vertices, skin triangles, eye triangles) using only the body
    and eye groups; clothing helpers, joints, teeth etc. are dropped."""
    vertices, skin, eyes, group = [], [], [], ""
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append(tuple(float(c) * DECIMETRE for c in parts[1:4]))
        elif parts[0] == "g":
            group = parts[1] if len(parts) > 1 else ""
        elif parts[0] == "f":
            target = skin if group == "body" else eyes if group in ("helper-l-eye", "helper-r-eye") else None
            if target is None:
                continue
            corners = [int(p.split("/")[0]) - 1 for p in parts[1:]]
            for k in range(1, len(corners) - 1):
                target.append((corners[0], corners[k], corners[k + 1]))
    return vertices, skin, eyes


def read_target(path):
    entries = []
    for line in open(path):
        parts = line.split()
        if len(parts) == 4 and not line.startswith("#"):
            entries.append((int(parts[0]), tuple(float(c) * DECIMETRE for c in parts[1:4])))
    return entries


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "target", "makehuman-cache")
    os.makedirs(cache, exist_ok=True)

    files = target_files()
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        base_path = pool.submit(fetch, cache, "3dobjs/base.obj")
        skeleton_path = pool.submit(fetch, cache, "rigs/default.mhskel")
        weights_path = pool.submit(fetch, cache, "rigs/default_weights.mhw")
        local = list(pool.map(lambda f: fetch(cache, f[1]), files))
    vertices, skin, eyes = read_base(base_path.result())
    skeleton = json.load(open(skeleton_path.result()))
    weights = json.load(open(weights_path.result()))["weights"]

    # Bones, parents first.
    bones, pending = [], dict(skeleton["bones"])
    while pending:
        for name in sorted(pending):
            parent = pending[name]["parent"]
            if parent is None or parent in bones:
                bones.append(name)
                del pending[name]
    joints = skeleton["joints"]

    # Keep only vertices the kept triangles use, then the joint markers,
    # and renumber.
    mesh_vertices = sorted({i for tri in skin + eyes for i in tri})
    in_mesh = set(mesh_vertices)
    markers = sorted({i for b in skeleton["bones"].values() for end in ("head", "tail") for i in joints[b[end]]} - in_mesh)
    used = mesh_vertices + markers
    remap = {old: new for new, old in enumerate(used)}
    triangles = [tuple(remap[i] for i in tri) for tri in skin + eyes]

    out = bytearray(b"C3BODY2\0")
    out += struct.pack("<II", len(used), len(mesh_vertices))
    for old in used:
        out += struct.pack("<3f", *vertices[old])
    out += struct.pack("<II", len(triangles), len(skin))
    for tri in triangles:
        out += struct.pack("<3I", *tri)
    out += struct.pack("<I", len(files))
    total = 0
    for (name, _), path in zip(files, local):
        entries = [(remap[i], d) for i, d in read_target(path) if i in remap]
        total += len(entries)
        encoded = name.encode()
        out += struct.pack("<H", len(encoded)) + encoded + struct.pack("<I", len(entries))
        for index, offset in entries:
            out += struct.pack("<I3f", index, *offset)

    out += struct.pack("<I", len(bones))
    for name in bones:
        bone = skeleton["bones"][name]
        encoded = name.encode()
        out += struct.pack("<H", len(encoded)) + encoded
        out += struct.pack("<i", bones.index(bone["parent"]) if bone["parent"] else -1)
        for end in ("head", "tail"):
            indices = [remap[i] for i in joints[bone[end]]]
            out += struct.pack(f"<I{len(indices)}I", len(indices), *indices)

    # The four strongest influences per vertex, renormalised.
    influences = {}
    for name, pairs in weights.items():
        for vertex, weight in pairs:
            if vertex in in_mesh and weight > 0:
                influences.setdefault(vertex, []).append((weight, bones.index(name)))
    unweighted = 0
    for old in mesh_vertices:
        top = sorted(influences.get(old, []), reverse=True)[:4]
        if not top:
            unweighted += 1
            top = [(1.0, 0)]
        weight_sum = sum(w for w, _ in top)
        top += [(0.0, 0)] * (4 - len(top))
        out += struct.pack("<4H", *[b for _, b in top]) + struct.pack("<4f", *[w / weight_sum for w, _ in top])
    print(f"{len(bones)} bones, {len(markers)} joint marker vertices, {unweighted} unweighted vertices")

    destination = os.path.join(root, "assets", "avatar", "body.bin")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        f.write(out)
    print(f"{len(used)} vertices, {len(skin)} skin + {len(eyes)} eye triangles, "
          f"{len(files)} targets ({total} offsets) -> {destination} ({len(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
