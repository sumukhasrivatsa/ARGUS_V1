"""
GraphViz.py — ARGUS Scene Graph Visualizer

Three-panel figure:
  LEFT   — top-down spatial map of the table (objects at real positions)
  CENTRE — rich knowledge graph (pastel nodes, colour-coded relation edges)
  RIGHT  — affordance weight bar chart + path analysis

No imports from scene_graph — no circular import.
"""

import math

ROBOT_REACH_RADIUS = 0.85   # keep in sync with scene_graph.py

# ── node palette (pastel, one colour per object, cycles) ──────────────────────
NODE_PALETTE = [
    "#FFB3C6", "#FFD6A5", "#FDFFB6", "#CAFFBF",
    "#9BF6FF", "#BDB2FF", "#FFC6FF", "#A0C4FF",
    "#FFADAD", "#D4E09B",
]

# ── edge colour by relation family ────────────────────────────────────────────
EDGE_FAMILY = {
    # path — red
    "on_path_to_goal":          "#E63946",
    "blocking_goal":            "#C1121F",
    # proximity — blue
    "near":                     "#4361EE",
    "close":                    "#4CC9F0",
    "moderate_distance":        "#ADB5BD",
    "far_from":                 "#CED4DA",
    # lateral — green
    "left_of":                  "#2DC653",
    "right_of":                 "#2DC653",
    "directly_left_of":         "#1A7431",
    "directly_right_of":        "#1A7431",
    # depth — teal
    "in_front_of":              "#0096C7",
    "behind":                   "#0096C7",
    "directly_ahead":           "#023E8A",
    "directly_behind":          "#023E8A",
    # vertical — orange
    "above":                    "#FB8500",
    "below":                    "#FB8500",
    "same_level":               "#FFB703",
    "stacked_on":               "#E85D04",
    # occlusion — purple
    "may_occlude":              "#7B2D8B",
    # robot-centric — brown/grey
    "closer_to_robot_than":     "#6D6875",
    "further_from_robot_than":  "#B5838D",
    "more_reachable_than":      "#457B9D",
    "less_reachable_than":      "#A8DADC",
}
DEFAULT_EDGE_COLOUR = "#999999"

PRIORITY_RELATIONS = [
    "blocking_goal", "on_path_to_goal", "stacked_on",
    "may_occlude", "near", "close",
    "directly_left_of", "directly_right_of",
    "directly_ahead", "directly_behind",
    "left_of", "right_of", "in_front_of", "behind",
    "above", "below", "same_level",
    "closer_to_robot_than", "further_from_robot_than",
    "far_from", "moderate_distance",
]


def _top_relations(relations, n=2):
    """Return up to n most important relations, in priority order."""
    result = []
    for p in PRIORITY_RELATIONS:
        if p in relations:
            result.append(p)
        if len(result) == n:
            break
    if not result and relations:
        result = list(relations[:n])
    return result


def _fragility_colour(f):
    """0.0 → green, 1.0 → red."""
    r = min(1.0, f * 2)
    g = min(1.0, (1 - f) * 2)
    return (r, g, 0.25, 0.88)


def visualize(
    graph,
    save_path: str = "/tmp/scene_graph_viz.png",
    show: bool = False,
) -> str:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("pip install matplotlib")
        return ""

    try:
        import networkx as nx
        HAS_NX = True
    except ImportError:
        HAS_NX = False

    path_obs  = set(graph.get_path_obstacles())
    goal_node = graph.nodes.get(graph.goal_label)
    labels    = list(graph.nodes.keys())
    n         = len(labels)

    # ── build edge map ────────────────────────────────────────────────────────
    edge_map = {}   # (src, tgt) → [relation, ...]
    for edge in graph.edges:
        edge_map[(edge.source, edge.target)] = edge.relations

    # ── layout ───────────────────────────────────────────────────────────────
    if HAS_NX:
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(labels)
        for (s, t) in edge_map:
            G.add_edge(s, t)
        raw = nx.spring_layout(G, k=2.8, seed=42, iterations=120)
        gpos = {l: (raw[l][0], raw[l][1]) for l in labels}
    else:
        gpos = {}
        for i, lb in enumerate(labels):
            a = 2 * math.pi * i / n
            gpos[lb] = (math.cos(a) * 0.65, math.sin(a) * 0.65)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE — 3 panels
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 9), facecolor="#F0F0F0")
    gs  = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.6, 2.0, 1.0],
        wspace=0.08,
        left=0.03, right=0.97,
        top=0.92, bottom=0.08,
    )

    ax_map   = fig.add_subplot(gs[0])   # top-down spatial map
    ax_graph = fig.add_subplot(gs[1])   # knowledge graph
    ax_bar   = fig.add_subplot(gs[2])   # weight chart

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 1 — top-down table map
    # ══════════════════════════════════════════════════════════════════════════
    ax_map.set_facecolor("#1C1C2E")
    ax_map.set_title("Spatial Map — Top Down", color="white",
                     fontsize=12, fontweight="bold", pad=8)

    # table
    table_patch = FancyBboxPatch(
        (-0.70, -0.45), 1.40, 0.90,
        boxstyle="round,pad=0.02",
        linewidth=2, edgecolor="#D4A843", facecolor="#2E1F0F", zorder=0
    )
    ax_map.add_patch(table_patch)

    # grid lines on table
    for gx in [-0.5, -0.25, 0.0, 0.25, 0.5]:
        ax_map.axvline(gx, color="#ffffff", alpha=0.04, linewidth=0.5, zorder=1)
    for gy in [-0.3, -0.15, 0.0, 0.15, 0.3]:
        ax_map.axhline(gy, color="#ffffff", alpha=0.04, linewidth=0.5, zorder=1)

    # robot
    rx, ry, _ = graph.robot_pos
    reach_c = plt.Circle((rx, ry), ROBOT_REACH_RADIUS,
                         color="#FFD166", fill=False,
                         linestyle="--", lw=0.8, alpha=0.25, zorder=1)
    ax_map.add_patch(reach_c)
    ax_map.plot(rx, ry, marker="*", color="#FFD166",
                markersize=16, zorder=5, label="robot")
    ax_map.text(rx + 0.03, ry + 0.06, "robot",
                color="#FFD166", fontsize=7, zorder=6)

    # camera
    cx, cy, _ = graph.camera_pos
    ax_map.plot(cx, cy, marker="s", color="#06D6A0",
                markersize=9, zorder=5)
    ax_map.text(cx + 0.03, cy + 0.06, "camera",
                color="#06D6A0", fontsize=7, zorder=6)

    # path robot → goal
    if goal_node:
        gx2, gy2, _ = goal_node.pos
        ax_map.annotate(
            "", xy=(gx2, gy2), xytext=(rx, ry),
            arrowprops=dict(
                arrowstyle="->", color="#FFB703", lw=1.8,
                linestyle="dashed", connectionstyle="arc3,rad=0.0",
            ), zorder=2,
        )

    # faint edges on map
    for edge in graph.edges:
        if edge.source not in graph.nodes or edge.target not in graph.nodes:
            continue
        ap = graph.nodes[edge.source].pos
        bp = graph.nodes[edge.target].pos
        ax_map.plot([ap[0], bp[0]], [ap[1], bp[1]],
                    color="#ffffff", alpha=0.07, lw=0.7, zorder=2)

    # objects
    for i, (label, node) in enumerate(graph.nodes.items()):
        x, y, _ = node.pos
        is_goal = (label == graph.goal_label)
        is_path = label in path_obs
        colour  = _fragility_colour(node.fragility)
        size    = 160 + node.conf * 240

        if is_path:
            ax_map.scatter(x, y, s=size * 2.4,
                           color="#E63946", alpha=0.28, zorder=3)
        if is_goal:
            ax_map.scatter(x, y, s=size * 2.8,
                           color="#FFD166", alpha=0.35,
                           marker="*", zorder=3)

        ax_map.scatter(x, y, s=size, color=colour,
                       edgecolors="white", linewidths=0.9, zorder=4)

        reach_ch = "✓" if node.reachable else "✗"
        ax_map.text(
            x, y + 0.07,
            f"{label}\n{reach_ch} f={node.fragility:.1f}",
            color="white", fontsize=6.5, ha="center", va="bottom",
            zorder=5,
            bbox=dict(facecolor="#00000077", edgecolor="none",
                      boxstyle="round,pad=0.15"),
        )

    ax_map.set_xlim(-0.85, 0.85)
    ax_map.set_ylim(-0.65, 0.65)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x (m)", color="#aaaaaa", fontsize=8)
    ax_map.set_ylabel("y (m)", color="#aaaaaa", fontsize=8)
    ax_map.tick_params(colors="#aaaaaa", labelsize=7)
    for sp in ax_map.spines.values():
        sp.set_edgecolor("#444444")

    map_legend = [
        mpatches.Patch(color="#2DC653", alpha=0.8, label="robust"),
        mpatches.Patch(color="#E63946", alpha=0.8, label="fragile"),
        mpatches.Patch(color="#E63946", alpha=0.3, label="on path"),
        mpatches.Patch(color="#FFD166", alpha=0.4, label="goal"),
    ]
    ax_map.legend(handles=map_legend, loc="lower right",
                  facecolor="#1C1C2E", edgecolor="#444",
                  labelcolor="white", fontsize=6.5)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 2 — knowledge graph
    # ══════════════════════════════════════════════════════════════════════════
    ax_graph.set_facecolor("#FAFAFA")
    ax_graph.set_title("Scene Graph — Spatial Relations",
                       color="#111111", fontsize=12,
                       fontweight="bold", pad=8)
    ax_graph.axis("off")

    all_gx = [p[0] for p in gpos.values()]
    all_gy = [p[1] for p in gpos.values()]
    pad2   = 0.45
    panel2 = FancyBboxPatch(
        (min(all_gx) - pad2, min(all_gy) - pad2),
        (max(all_gx) - min(all_gx)) + pad2 * 2,
        (max(all_gy) - min(all_gy)) + pad2 * 2,
        boxstyle="round,pad=0.05",
        linewidth=1.2, edgecolor="#DDDDDD",
        facecolor="#FFFFFF", zorder=0,
    )
    ax_graph.add_patch(panel2)

    NODE_R = 0.14

    # draw edges first (behind nodes)
    for (src, tgt), relations in edge_map.items():
        if src not in gpos or tgt not in gpos:
            continue
        sx, sy = gpos[src]
        tx, ty = gpos[tgt]
        dx, dy = tx - sx, ty - sy
        dist   = math.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            continue
        ux, uy = dx / dist, dy / dist

        start = (sx + ux * NODE_R, sy + uy * NODE_R)
        end   = (tx - ux * NODE_R * 1.1, ty - uy * NODE_R * 1.1)

        top3    = _top_relations(relations, n=3)
        primary = top3[0] if top3 else ""
        ecol    = EDGE_FAMILY.get(primary, DEFAULT_EDGE_COLOUR)

        ax_graph.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="-|>",
                color=ecol, lw=1.4,
                mutation_scale=14,
                connectionstyle="arc3,rad=0.10",
            ),
            zorder=2,
        )

        # edge label box — each relation on its own line,
        # coloured by its own family so colour AND text are both present
        mx   = (start[0] + end[0]) / 2
        my   = (start[1] + end[1]) / 2
        perp = 0.10
        px   = -uy * perp
        py   =  ux * perp

        # build one text string but we draw each line separately so we can
        # give each its own colour — stack them vertically
        line_h = 0.055   # vertical gap between lines
        n_rels = len(top3)
        for k, rel in enumerate(top3):
            rcol  = EDGE_FAMILY.get(rel, DEFAULT_EDGE_COLOUR)
            yoff  = (n_rels - 1) / 2 * line_h - k * line_h
            rtext = rel.replace("_", " ")
            ax_graph.text(
                mx + px, my + py + yoff,
                rtext,
                fontsize=6.5, color=rcol,
                ha="center", va="center",
                fontweight="bold",
                zorder=5,
                bbox=dict(
                    facecolor="white",
                    edgecolor=rcol,
                    boxstyle="round,pad=0.14",
                    linewidth=0.9,
                    alpha=0.93,
                ) if k == 0 else dict(   # only box the primary relation
                    facecolor="white",
                    edgecolor="none",
                    boxstyle="round,pad=0.14",
                    alpha=0.85,
                ),
            )

    # draw nodes
    for i, label in enumerate(labels):
        x, y    = gpos[label]
        node    = graph.nodes[label]
        is_goal = (label == graph.goal_label)
        is_path = label in path_obs
        colour  = "#FF6B6B" if is_goal else NODE_PALETTE[i % len(NODE_PALETTE)]
        bwidth  = 1.0 + node.fragility * 3.5
        bedge   = "#CC2222" if is_goal else (
            "#FF4444" if is_path else
            _fragility_colour(node.fragility)[:3]
        )

        # shadow
        shadow = plt.Circle((x + 0.015, y - 0.015), NODE_R,
                             color="#CCCCCC", zorder=2, alpha=0.5)
        ax_graph.add_patch(shadow)

        # node circle
        circ = plt.Circle((x, y), NODE_R, zorder=3)
        circ.set_facecolor(colour)
        circ.set_edgecolor(bedge)
        circ.set_linewidth(bwidth)
        ax_graph.add_patch(circ)

        # label inside node
        ax_graph.text(
            x, y + 0.018,
            label,
            ha="center", va="center",
            fontsize=8,
            fontweight="bold" if is_goal else "semibold",
            color="#CC0000" if is_goal else "#111111",
            zorder=5,
        )

        # confidence score inside node
        ax_graph.text(
            x, y - 0.045,
            f"{node.conf:.2f}",
            ha="center", va="center",
            fontsize=6, color="#555555",
            zorder=5,
        )

        # badges below node
        reach_ch  = "✓ reach" if node.reachable else "✗ unreach"
        frag_text = f"fragility {node.fragility:.1f}"
        cats      = " · ".join(node.categories[:2]) if node.categories else ""
        badge     = f"{reach_ch}   {frag_text}"
        if cats:
            badge += f"\n{cats}"

        ax_graph.text(
            x, y - NODE_R - 0.04,
            badge,
            ha="center", va="top",
            fontsize=5.5, color="#444444",
            zorder=5,
            bbox=dict(facecolor="#F5F5F5", edgecolor="#DDDDDD",
                      boxstyle="round,pad=0.2", linewidth=0.5),
        )

    ax_graph.set_aspect("equal")
    m3 = 0.55
    ax_graph.set_xlim(min(all_gx) - m3, max(all_gx) + m3)
    ax_graph.set_ylim(min(all_gy) - m3, max(all_gy) + m3)

    # edge colour legend
    family_legend = [
        mpatches.Patch(color="#E63946", label="path / blocking"),
        mpatches.Patch(color="#4361EE", label="proximity"),
        mpatches.Patch(color="#2DC653", label="lateral (L/R)"),
        mpatches.Patch(color="#0096C7", label="depth (front/behind)"),
        mpatches.Patch(color="#FB8500", label="vertical"),
        mpatches.Patch(color="#7B2D8B", label="may occlude"),
    ]
    ax_graph.legend(
        handles=family_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3, fontsize=6.5,
        framealpha=0.9, edgecolor="#CCCCCC",
        title="Edge colour = relation family",
        title_fontsize=6.5,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 3 — affordance weight bar chart
    # ══════════════════════════════════════════════════════════════════════════
    ax_bar.set_facecolor("#FAFAFA")
    ax_bar.set_title("Suggested Affordance Weights",
                     color="#111111", fontsize=11,
                     fontweight="bold", pad=8)

    suggested = graph.suggest_default_weights()
    # include the goal
    if graph.goal_label and graph.goal_label in graph.nodes:
        suggested[graph.goal_label] = +200.0
    sorted_items = sorted(suggested.items(), key=lambda x: x[1], reverse=True)
    bar_labels   = [it[0] for it in sorted_items]
    bar_values   = [it[1] for it in sorted_items]

    def bar_colour(v, label):
        if v > 0:
            return "#FFD166"    # gold = goal
        if label in path_obs:
            return "#E63946"    # red = on path
        if v <= -600:
            return "#6D6875"    # dark = hard obstacle
        return "#4CC9F0"        # blue = soft obstacle

    colours = [bar_colour(v, l) for l, v in zip(bar_labels, bar_values)]

    bars = ax_bar.barh(bar_labels, bar_values,
                       color=colours, edgecolor="#CCCCCC",
                       linewidth=0.6, height=0.6)

    # value labels on bars
    for bar, val in zip(bars, bar_values):
        xa = bar.get_width()
        xoff = 12 if xa >= 0 else -12
        align = "left" if xa >= 0 else "right"
        ax_bar.text(
            xa + xoff, bar.get_y() + bar.get_height() / 2,
            f"{int(val):+d}",
            va="center", ha=align, fontsize=7.5, color="#333333",
        )

    ax_bar.axvline(0, color="#888888", lw=0.8)
    ax_bar.set_xlabel("weight", fontsize=8, color="#444444")
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.tick_params(axis="x", labelsize=7)
    ax_bar.set_facecolor("#FAFAFA")
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#DDDDDD")

    # path analysis text box
    path_list = graph.get_path_obstacles()
    fragile   = graph.get_fragile_objects(0.6)
    info_lines = []
    if path_list:
        info_lines.append(f"⚠ On path: {', '.join(path_list)}")
    if fragile:
        info_lines.append(f"⚡ Fragile: {', '.join(fragile)}")
    if graph.goal_label:
        info_lines.append(f"🎯 Goal: {graph.goal_label}")
    clusters = graph.get_cluster_summary()
    for cl in clusters:
        info_lines.append(f"📍 Cluster: {', '.join(cl)}")

    if info_lines:
        ax_bar.text(
            0.5, -0.22, "\n".join(info_lines),
            transform=ax_bar.transAxes,
            ha="center", va="top",
            fontsize=7, color="#333333",
            bbox=dict(facecolor="#F0F0F0", edgecolor="#CCCCCC",
                      boxstyle="round,pad=0.4"),
        )

    bar_legend = [
        mpatches.Patch(color="#FFD166", label="Goal"),
        mpatches.Patch(color="#E63946", label="On path (high priority)"),
        mpatches.Patch(color="#6D6875", label="Hard obstacle"),
        mpatches.Patch(color="#4CC9F0", label="Soft obstacle"),
    ]
    ax_bar.legend(handles=bar_legend, loc="lower right",
                  fontsize=6.5, framealpha=0.9, edgecolor="#CCCCCC")

    # ── main title ────────────────────────────────────────────────────────────
    n_obj = len(graph.nodes)
    n_rel = sum(len(e.relations) for e in graph.edges)
    fig.suptitle(
        f"ARGUS  ·  Scene Graph  ·  {n_obj} objects  ·  {n_rel} relations",
        fontsize=15, fontweight="bold", color="#111111", y=0.98,
    )

    plt.savefig(save_path, dpi=160, bbox_inches="tight",
                facecolor="#F0F0F0")
    if show:
        plt.show()
    plt.close()

    print(f"Scene graph saved → {save_path}")
    return save_path