"""Рендер DAG'а dbt-проекта в PNG из target/manifest.json (офлайн-визуализация графа)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_dbt_graph(manifest_path: Path, out_png: Path) -> dict[str, int]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    nodes: dict[str, dict] = {}
    for uid, n in manifest.get("nodes", {}).items():
        if n.get("resource_type") in ("model", "snapshot", "seed"):
            nodes[uid] = {"name": n["name"], "kind": n["resource_type"]}
    for uid, s in manifest.get("sources", {}).items():
        nodes[uid] = {"name": f"{s['source_name']}.{s['name']}", "kind": "source"}
    edges: list[tuple[str, str]] = []
    for uid, n in nodes.items():
        dep = manifest["nodes"].get(uid, manifest["sources"].get(uid, {}))
        for parent in dep.get("depends_on", {}).get("nodes", []):
            if parent in nodes:
                edges.append((parent, uid))

    # слои по топологической глубине
    depth: dict[str, int] = {}
    for _ in range(len(nodes) + 1):
        changed = False
        for child in nodes:
            parents = [p for p, c in edges if c == child]
            d = 0 if not parents else max(depth.get(p, 0) for p in parents) + 1
            if depth.get(child) != d:
                depth[child] = d
                changed = True
        if not changed:
            break
    by_layer: dict[int, list[str]] = {}
    for uid, d in depth.items():
        by_layer.setdefault(d, []).append(uid)

    fig, ax = plt.subplots(figsize=(9, 1 + 1.3 * max(by_layer or {0: []})))
    pos: dict[str, tuple[float, float]] = {}
    colors = {"source": "#e07a5f", "model": "#4878b0", "snapshot": "#54b683", "seed": "#8a8a8a"}
    for layer, uids in sorted(by_layer.items()):
        for i, uid in enumerate(sorted(uids, key=lambda u: nodes[u]["name"])):
            x = (i + 0.5) / len(uids)
            y = -layer
            pos[uid] = (x, y)
    for p, c in edges:
        ax.annotate("", xy=pos[c], xytext=pos[p],
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.1))
    for uid, (x, y) in pos.items():
        ax.scatter([x], [y], s=1400, color=colors.get(nodes[uid]["kind"], "#ccc"), zorder=3)
        ax.text(x, y, nodes[uid]["name"], ha="center", va="center", fontsize=7,
                color="white", zorder=4)
    ax.set_xlim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title("dbt DAG: raw → staging → marts (+snapshot)", fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return {"nodes": len(nodes), "edges": len(edges)}
