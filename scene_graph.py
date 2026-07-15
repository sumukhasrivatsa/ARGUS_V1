"""
scene_graph.py — ARGUS v1 Scene Graph Engine

Builds a rich, queryable 3D spatial scene graph from YOLO detections
and world-frame 3D positions. Designed to feed an LLM with structured
scene context so it can reason about unmentioned objects, spatial
relationships, reachability, and fragility.

No ML models required. No point clouds. No external graph libraries.
Runs entirely from YOLO labels + 3D positions your perception node
already produces.

Architecture:
    SceneGraphBuilder  → builds graph from raw detections each frame
    SceneGraph         → queryable graph with rich spatial relations
    SceneGraphDiff     → what changed between two frames
    LLMPromptBuilder   → formats graph for LLM consumption

Relation types computed:
    Proximity:     near, close, moderate_distance, far
    Lateral:       left_of, right_of, directly_left, directly_right
    Depth:         in_front_of, behind, directly_ahead, directly_behind
    Vertical:      above, below, same_level, stacked_on
    Robot-centric: closest_to_robot, furthest_from_robot, reachable, unreachable
    Path:          on_path_to_goal, blocking_path, clear_of_path
    Cluster:       co_located (tight group), isolated
    Occlusion:     may_occlude (estimated from camera angle)
"""

import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from GraphViz import visualize

# ─────────────────────────────────────────────────────────────────────────────
# Semantic knowledge base
# Encodes world knowledge about object categories that the LLM should not
# have to infer from scratch every time.
# ─────────────────────────────────────────────────────────────────────────────

FRAGILITY_SCORES = {
    # 1.0 = extremely fragile, 0.0 = indestructible
    "vase":         1.0,
    "wine glass":   1.0,
    "glass":        1.0,
    "bottle":       0.8,
    "cup":          0.6,
    "mug":          0.5,
    "bowl":         0.5,
    "laptop":       0.9,
    "cell phone":   0.8,
    "phone":        0.8,
    "clock":        0.7,
    "book":         0.2,
    "banana":       0.4,
    "apple":        0.3,
    "sports ball":  0.1,
    "red ball":     0.1,
    "ball":         0.1,
    "boot":         0.05,
    "shoe":         0.05,
    "hammer":       0.0,
    "knife":        0.3,
    "can":          0.2,
}

SEMANTIC_CATEGORIES = {
    "drinkware":  ["cup", "mug", "wine glass", "glass", "bottle", "can"],
    "fragile":    ["vase", "wine glass", "glass", "bottle", "laptop", "cell phone", "clock"],
    "food":       ["banana", "apple", "orange", "sandwich", "pizza", "donut", "cake"],
    "tool":       ["hammer", "knife", "scissors", "wrench"],
    "electronic": ["laptop", "cell phone", "phone", "clock", "keyboard", "mouse", "tv"],
    "container":  ["bowl", "cup", "mug", "bottle", "vase", "can"],
    "ball":       ["sports ball", "red ball", "ball", "tennis ball", "baseball"],
    "clothing":   ["boot", "shoe", "sneaker", "sandal"],
}

# Approximate real-world dimensions (metres) for size estimation
TYPICAL_SIZES = {
    "bottle":       (0.08, 0.30, 0.08),
    "vase":         (0.12, 0.25, 0.12),
    "cup":          (0.08, 0.10, 0.08),
    "mug":          (0.09, 0.10, 0.09),
    "bowl":         (0.18, 0.07, 0.18),
    "laptop":       (0.35, 0.03, 0.25),
    "sports ball":  (0.16, 0.16, 0.16),
    "red ball":     (0.16, 0.16, 0.16),
    "ball":         (0.16, 0.16, 0.16),
    "banana":       (0.20, 0.05, 0.05),
    "boot":         (0.30, 0.28, 0.12),
    "knife":        (0.22, 0.02, 0.02),
    "hammer":       (0.35, 0.06, 0.06),
    "phone":        (0.08, 0.01, 0.16),
    "clock":        (0.18, 0.22, 0.10),
}

# UR5e workspace limits (metres from robot base)
ROBOT_REACH_RADIUS = 0.85   # maximum reach
ROBOT_MIN_REACH    = 0.10   # minimum reach (too close)
ROBOT_BASE_HEIGHT  = 0.0    # z of robot base in ROS frame


# ─────────────────────────────────────────────────────────────────────────────
# Core data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Raw output from YOLO-World + depth projection."""
    label:    str
    pos:      tuple[float, float, float]   # (x, y, z) world frame metres
    conf:     float                         # YOLO confidence 0-1
    bbox_px:  tuple[int, int, int, int] = (0, 0, 0, 0)   # (x1,y1,x2,y2) pixels
    size_est: tuple[float, float, float] = (0.0, 0.0, 0.0)  # estimated (w,h,d)


@dataclass
class SceneNode:
    """A node in the scene graph — one detected object."""
    label:        str
    pos:          tuple[float, float, float]
    conf:         float
    size:         tuple[float, float, float]
    fragility:    float            # 0.0 (robust) → 1.0 (extremely fragile)
    categories:   list[str]        # e.g. ["fragile", "drinkware", "container"]
    reachable:    bool             # can the robot arm reach it
    dist_robot:   float            # distance from robot base (metres)
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "label":      self.label,
            "position":   {"x": round(self.pos[0], 3),
                           "y": round(self.pos[1], 3),
                           "z": round(self.pos[2], 3)},
            "confidence": round(self.conf, 3),
            "fragility":  self.fragility,
            "categories": self.categories,
            "reachable":  self.reachable,
            "dist_robot": round(self.dist_robot, 3),
        }


@dataclass
class SceneEdge:
    """A directed edge representing spatial relations between two objects."""
    source:    str
    target:    str
    relations: list[str]
    dist_2d:   float    # horizontal distance metres
    dist_3d:   float    # 3D Euclidean distance metres

    def to_dict(self) -> dict:
        return {
            "from":      self.source,
            "to":        self.target,
            "relations": self.relations,
            "dist_2d":   round(self.dist_2d, 3),
            "dist_3d":   round(self.dist_3d, 3),
        }


@dataclass
class SceneGraphDiff:
    """What changed between two consecutive scene graph snapshots."""
    appeared:   list[str]                      # new objects
    disappeared: list[str]                     # objects no longer detected
    moved:      dict[str, tuple[float, float]] # label → (old_pos, new_pos, delta_m)
    timestamp:  float = field(default_factory=time.time)

    def is_significant(self, move_threshold_m: float = 0.05) -> bool:
        if self.appeared or self.disappeared:
            return True
        return any(v[2] > move_threshold_m for v in self.moved.values())

    def to_natural_language(self) -> str:
        lines = []
        if self.appeared:
            lines.append(f"New objects detected: {', '.join(self.appeared)}")
        if self.disappeared:
            lines.append(f"Objects no longer visible: {', '.join(self.disappeared)}")
        for label, (old, new, delta) in self.moved.items():
            lines.append(
                f"{label} moved {delta*100:.1f}cm "
                f"(from ({old[0]:.2f},{old[1]:.2f}) "
                f"to ({new[0]:.2f},{new[1]:.2f}))"
            )
        return "\n".join(lines) if lines else "No significant changes."


# ─────────────────────────────────────────────────────────────────────────────
# Scene Graph
# ─────────────────────────────────────────────────────────────────────────────

class SceneGraph:
    """
    Queryable 3D spatial scene graph.

    Stores nodes (objects) and directed edges (spatial relations).
    Provides a rich query API and LLM-ready serialization.
    """

    def __init__(
        self,
        nodes: dict[str, SceneNode],
        edges: list[SceneEdge],
        robot_pos: tuple[float, float, float],
        goal_label: Optional[str],
        camera_pos: tuple[float, float, float],
        timestamp: float,
    ):
        self.nodes      = nodes
        self.edges      = edges
        self.robot_pos  = robot_pos
        self.goal_label = goal_label
        self.camera_pos = camera_pos
        self.timestamp  = timestamp

        # adjacency index for fast lookups
        self._adj: dict[str, list[SceneEdge]] = {label: [] for label in nodes}
        for edge in edges:
            self._adj[edge.source].append(edge)
            # also index reverse direction
            rev = SceneEdge(
                source=edge.target,
                target=edge.source,
                relations=self._reverse_relations(edge.relations),
                dist_2d=edge.dist_2d,
                dist_3d=edge.dist_3d,
            )
            self._adj[edge.target].append(rev)

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_node(self, label: str) -> Optional[SceneNode]:
        return self.nodes.get(label)

    def get_edges(self, label: str) -> list[SceneEdge]:
        """All edges involving this object."""
        return self._adj.get(label, [])

    def get_relations(self, source: str, target: str) -> list[str]:
        """Get all spatial relations from source to target."""
        for edge in self._adj.get(source, []):
            if edge.target == target:
                return edge.relations
        return []

    def get_objects_with_relation(self, label: str, relation: str) -> list[str]:
        """Find all objects that have the given relation to label."""
        result = []
        for edge in self._adj.get(label, []):
            if relation in edge.relations:
                result.append(edge.target)
        return result

    def get_fragile_objects(self, threshold: float = 0.5) -> list[str]:
        """Return labels of objects above fragility threshold."""
        return [l for l, n in self.nodes.items() if n.fragility >= threshold]

    def get_reachable_objects(self) -> list[str]:
        """Return labels of objects the robot arm can reach."""
        return [l for l, n in self.nodes.items() if n.reachable]

    def get_path_obstacles(self) -> list[str]:
        """
        Return labels of objects on the path from robot to goal.
        These are the ones the planner should prioritize avoiding.
        """
        if self.goal_label is None:
            return []
        result = []
        for label in self.nodes:
            if label == self.goal_label:
                continue
            for edge in self._adj.get(label, []):
                if "on_path_to_goal" in edge.relations or \
                   "blocking_goal" in edge.relations:
                    result.append(label)
                    break
        return list(set(result))

    def get_objects_by_category(self, category: str) -> list[str]:
        """Return labels of objects belonging to a semantic category."""
        return [l for l, n in self.nodes.items() if category in n.categories]

    def get_nearest_to(self, label: str, k: int = 3) -> list[tuple[str, float]]:
        """Return k nearest objects to label, sorted by distance."""
        node = self.nodes.get(label)
        if node is None:
            return []
        distances = []
        for other_label, other_node in self.nodes.items():
            if other_label == label:
                continue
            dist = _dist_3d(node.pos, other_node.pos)
            distances.append((other_label, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def get_cluster_summary(self) -> list[list[str]]:
        """
        Group objects into spatial clusters (objects within 0.15m of each other).
        Returns list of clusters, each cluster being a list of labels.
        """
        labels = list(self.nodes.keys())
        visited = set()
        clusters = []

        for label in labels:
            if label in visited:
                continue
            cluster = [label]
            visited.add(label)
            node = self.nodes[label]
            for other_label, other_node in self.nodes.items():
                if other_label in visited:
                    continue
                if _dist_2d(node.pos, other_node.pos) < 0.15:
                    cluster.append(other_label)
                    visited.add(other_label)
            clusters.append(cluster)

        return [c for c in clusters if len(c) > 1]  # only multi-object clusters

    def get_sorted_by_distance_to_robot(self) -> list[tuple[str, float]]:
        """All objects sorted by distance to robot base, nearest first."""
        pairs = [(l, n.dist_robot) for l, n in self.nodes.items()]
        return sorted(pairs, key=lambda x: x[1])

    def suggest_default_weights(self) -> dict[str, float]:
        """
        Suggest affordance weights for objects not mentioned by the user.
        Uses fragility, path position, reachability, and proximity to goal.
        """
        weights = {}
        goal_node = self.nodes.get(self.goal_label) if self.goal_label else None
        path_obstacles = set(self.get_path_obstacles())

        for label, node in self.nodes.items():
            if label == self.goal_label:
                continue

            base = -200.0  # default moderate avoid

            # fragility increases avoidance priority
            base -= node.fragility * 600.0

            # on path to goal → much higher avoidance
            if label in path_obstacles:
                base -= 200.0

            # very close to goal → higher avoidance (don't knock it over reaching)
            if goal_node:
                dist_to_goal = _dist_2d(node.pos, goal_node.pos)
                if dist_to_goal < 0.15:
                    base -= 150.0

            # unreachable objects don't matter much
            if not node.reachable:
                base = max(base, -100.0)

            weights[label] = round(max(-1000.0, base))

        return weights

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Full JSON-serializable representation."""
        return {
            "timestamp":   self.timestamp,
            "robot_pos":   {"x": self.robot_pos[0], "y": self.robot_pos[1], "z": self.robot_pos[2]},
            "goal":        self.goal_label,
            "nodes":       {l: n.to_dict() for l, n in self.nodes.items()},
            "edges":       [e.to_dict() for e in self.edges],
            "clusters":    self.get_cluster_summary(),
            "path_obstacles": self.get_path_obstacles(),
            "fragile_objects": self.get_fragile_objects(),
            "reachable_objects": self.get_reachable_objects(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_prompt_string(self, include_weights_hint: bool = True) -> str:
        """
        Formats the complete scene graph as a natural language string
        ready to be injected into an LLM prompt.
        Ordered to give the LLM the most useful context first.
        """
        lines = []
        lines.append("╔══════════════════════════════╗")
        lines.append("║     3D SCENE GRAPH           ║")
        lines.append("╚══════════════════════════════╝")
        lines.append("")

        rx, ry, rz = self.robot_pos
        lines.append(f"Robot base: ({rx:.2f}, {ry:.2f}, {rz:.2f})m  |  Max reach: {ROBOT_REACH_RADIUS}m")
        cx, cy, cz = self.camera_pos
        lines.append(f"Camera:     ({cx:.2f}, {cy:.2f}, {cz:.2f})m")
        if self.goal_label:
            gn = self.nodes.get(self.goal_label)
            if gn:
                lines.append(f"Goal:       {self.goal_label} at ({gn.pos[0]:.2f}, {gn.pos[1]:.2f}, {gn.pos[2]:.2f})m")
        lines.append("")

        # objects table
        lines.append("── OBJECTS (" + str(len(self.nodes)) + " detected) ──")
        sorted_objs = self.get_sorted_by_distance_to_robot()
        for label, dist in sorted_objs:
            node = self.nodes[label]
            x, y, z = node.pos
            reach_str  = "✓ reachable" if node.reachable else "✗ unreachable"
            frag_str   = f"fragility={node.fragility:.1f}"
            cat_str    = "/".join(node.categories[:2]) if node.categories else "misc"
            lines.append(
                f"  {label:20s} pos=({x:+.2f},{y:+.2f},{z:+.2f})m  "
                f"dist={dist:.2f}m  conf={node.conf:.2f}  "
                f"{frag_str}  [{cat_str}]  {reach_str}"
            )
        lines.append("")

        # spatial relations
        lines.append("── SPATIAL RELATIONS ──")
        for edge in self.edges:
            rel_str = ", ".join(edge.relations)
            lines.append(
                f"  {edge.source:20s} → {edge.target:20s}  [{rel_str}]  "
                f"dist={edge.dist_2d:.2f}m"
            )
        lines.append("")

        # path analysis
        path_obs = self.get_path_obstacles()
        if path_obs:
            lines.append(f"── PATH ANALYSIS ──")
            lines.append(f"  Objects on path to goal: {', '.join(path_obs)}")
            lines.append(f"  (These must be avoided to reach the goal)")
            lines.append("")

        # clusters
        clusters = self.get_cluster_summary()
        if clusters:
            lines.append("── OBJECT CLUSTERS ──")
            for cluster in clusters:
                lines.append(f"  Tight group: {', '.join(cluster)} (within 15cm)")
            lines.append("")

        # fragility warning
        fragile = self.get_fragile_objects(0.5)
        if fragile:
            lines.append(f"── FRAGILE OBJECTS (handle with care) ──")
            for label in fragile:
                lines.append(
                    f"  {label}: fragility={self.nodes[label].fragility:.1f}")
            lines.append("")

        if include_weights_hint:
            suggested = self.suggest_default_weights()
            lines.append("── SUGGESTED DEFAULT WEIGHTS (unmentioned objects) ──")
            lines.append("  (Based on fragility, path position, proximity to goal)")
            for label, w in sorted(suggested.items(), key=lambda x: x[1]):
                lines.append(f"  {label:20s}: {w:+.0f}")
            lines.append("")

        lines.append("══════════════════════════════")
        return "\n".join(lines)

    def _reverse_relations(self, relations: list[str]) -> list[str]:
        """Flip directional relations for reverse edge indexing."""
        mapping = {
            "left_of": "right_of",
            "right_of": "left_of",
            "directly_left": "directly_right",
            "directly_right": "directly_left",
            "in_front_of": "behind",
            "behind": "in_front_of",
            "directly_ahead": "directly_behind",
            "directly_behind": "directly_ahead",
            "above": "below",
            "below": "above",
            "closer_to_robot_than": "further_from_robot_than",
            "further_from_robot_than": "closer_to_robot_than",
        }
        return [mapping.get(r, r) for r in relations]


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

class SceneGraphBuilder:
    """
    Builds SceneGraph objects from raw YOLO detections each perception cycle.

    Also maintains the previous graph snapshot so it can produce a
    SceneGraphDiff describing what changed between frames.
    """

    # Tunable thresholds
    NEAR_M           = 0.15   # "near" proximity threshold
    CLOSE_M          = 0.25   # "close" proximity threshold
    FAR_M            = 0.45   # "far" threshold
    SAME_LEVEL_Z_TOL = 0.06   # z tolerance for "same_level"
    LATERAL_THRESH   = 0.07   # min x/y diff to claim left/right or front/back
    DIRECT_ALIGN_TOL = 0.06   # tolerance for "directly" aligned
    PATH_TOL_M       = 0.13   # how close to robot→goal line counts as "on_path"
    OCCLUDE_ANGLE    = 15.0   # degrees, angular threshold for occlusion estimate

    def __init__(
        self,
        robot_pos:  tuple[float, float, float] = (-0.5, 0.0, 0.0),
        camera_pos: tuple[float, float, float] = (0.5, -1.4, 0.825),
    ):
        self.robot_pos  = robot_pos
        self.camera_pos = camera_pos
        self._prev_graph: Optional[SceneGraph] = None

    def build(
        self,
        detections: list[Detection],
        goal_label: Optional[str] = None,
    ) -> tuple["SceneGraph", Optional[SceneGraphDiff]]:
        """
        Build a new SceneGraph from raw detections.
        Returns (graph, diff) where diff is None on the first call.
        """
        nodes = self._build_nodes(detections)
        edges = self._build_edges(nodes, goal_label)

        graph = SceneGraph(
            nodes=nodes,
            edges=edges,
            robot_pos=self.robot_pos,
            goal_label=goal_label,
            camera_pos=self.camera_pos,
            timestamp=time.time(),
        )

        diff = self._compute_diff(self._prev_graph, graph)
        self._prev_graph = graph
        return graph, diff

    def _build_nodes(self, detections: list[Detection]) -> dict[str, SceneNode]:
        nodes = {}
        for det in detections:
            label   = det.label
            pos     = det.pos
            dist    = _dist_3d(pos, self.robot_pos)
            reach   = ROBOT_MIN_REACH < dist < ROBOT_REACH_RADIUS and \
                      pos[2] >= -0.1   # not underground

            # look up or estimate size
            size = TYPICAL_SIZES.get(label, (0.10, 0.15, 0.10))
            if det.size_est != (0.0, 0.0, 0.0):
                size = det.size_est

            # semantic enrichment
            fragility = FRAGILITY_SCORES.get(label, 0.3)
            categories = [
                cat for cat, members in SEMANTIC_CATEGORIES.items()
                if label in members
            ]

            nodes[label] = SceneNode(
                label=label,
                pos=pos,
                conf=det.conf,
                size=size,
                fragility=fragility,
                categories=categories,
                reachable=reach,
                dist_robot=dist,
            )
        return nodes

    def _build_edges(
        self,
        nodes: dict[str, SceneNode],
        goal_label: Optional[str],
    ) -> list[SceneEdge]:
        edges = []
        labels = list(nodes.keys())
        goal_pos = nodes[goal_label].pos if goal_label and goal_label in nodes else None

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a_lbl = labels[i]
                b_lbl = labels[j]
                a     = nodes[a_lbl]
                b     = nodes[b_lbl]

                relations = self._compute_relations(a, b, goal_pos)

                if relations:
                    edges.append(SceneEdge(
                        source=a_lbl,
                        target=b_lbl,
                        relations=relations,
                        dist_2d=_dist_2d(a.pos, b.pos),
                        dist_3d=_dist_3d(a.pos, b.pos),
                    ))

        return edges

    def _compute_relations(
        self,
        a: SceneNode,
        b: SceneNode,
        goal_pos: Optional[tuple],
    ) -> list[str]:
        relations = []
        ax, ay, az = a.pos
        bx, by, bz = b.pos
        d2  = _dist_2d(a.pos, b.pos)
        d3  = _dist_3d(a.pos, b.pos)
        dx  = bx - ax
        dy  = by - ay
        dz  = bz - az

        # ── proximity ─────────────────────────────────────────────────────────
        if d2 < self.NEAR_M:
            relations.append("near")
        elif d2 < self.CLOSE_M:
            relations.append("close")
        elif d2 > self.FAR_M:
            relations.append("far_from")
        else:
            relations.append("moderate_distance")

        # ── lateral (x-axis) ──────────────────────────────────────────────────
        if abs(dx) > self.LATERAL_THRESH:
            tag = "directly_" if abs(dy) < self.DIRECT_ALIGN_TOL else ""
            if dx > 0:
                relations.append(f"{tag}left_of")
            else:
                relations.append(f"{tag}right_of")

        # ── depth (y-axis, camera faces +y) ──────────────────────────────────
        if abs(dy) > self.LATERAL_THRESH:
            tag = "directly_" if abs(dx) < self.DIRECT_ALIGN_TOL else ""
            if dy > 0:
                relations.append(f"{tag}in_front_of")
            else:
                relations.append(f"{tag}behind")

        # ── vertical (z-axis) ─────────────────────────────────────────────────
        if abs(dz) < self.SAME_LEVEL_Z_TOL:
            relations.append("same_level")
        elif dz > self.SAME_LEVEL_Z_TOL:
            relations.append("below")
            if dz < a.size[1] * 0.5:
                relations.append("stacked_on")
        else:
            relations.append("above")

        # ── robot-centric ─────────────────────────────────────────────────────
        da = _dist_2d(a.pos, self.robot_pos)
        db = _dist_2d(b.pos, self.robot_pos)
        if abs(da - db) > 0.10:
            if da < db:
                relations.append("closer_to_robot_than")
            else:
                relations.append("further_from_robot_than")

        # ── reachability comparison ────────────────────────────────────────────
        if a.reachable and not b.reachable:
            relations.append("more_reachable_than")
        elif b.reachable and not a.reachable:
            relations.append("less_reachable_than")

        # ── path analysis ─────────────────────────────────────────────────────
        if goal_pos is not None:
            a_on_path = _is_on_path(a.pos, self.robot_pos, goal_pos, self.PATH_TOL_M)
            b_on_path = _is_on_path(b.pos, self.robot_pos, goal_pos, self.PATH_TOL_M)

            if a_on_path:
                relations.append("on_path_to_goal")
                # is it also close to the goal? → likely blocking it
                if _dist_2d(a.pos, goal_pos) < 0.20:
                    relations.append("blocking_goal")
            if b_on_path:
                relations.append("target_on_path_to_goal")

            # are both clear of the path?
            if not a_on_path and not b_on_path:
                relations.append("clear_of_path")

        # ── occlusion estimate (from camera perspective) ───────────────────────
        # If a is between the camera and b (same angle, a is closer), a may occlude b
        cam = self.camera_pos
        angle_a = math.degrees(math.atan2(a.pos[0]-cam[0], a.pos[1]-cam[1]))
        angle_b = math.degrees(math.atan2(b.pos[0]-cam[0], b.pos[1]-cam[1]))
        dist_cam_a = _dist_2d(a.pos, cam)
        dist_cam_b = _dist_2d(b.pos, cam)

        if abs(angle_a - angle_b) < self.OCCLUDE_ANGLE and dist_cam_a < dist_cam_b:
            relations.append("may_occlude")

        return relations

    def _compute_diff(
        self,
        prev: Optional[SceneGraph],
        curr: SceneGraph,
        move_threshold_m: float = 0.04,
    ) -> Optional[SceneGraphDiff]:
        if prev is None:
            return None

        prev_labels = set(prev.nodes.keys())
        curr_labels = set(curr.nodes.keys())

        appeared    = list(curr_labels - prev_labels)
        disappeared = list(prev_labels - curr_labels)
        moved       = {}

        for label in prev_labels & curr_labels:
            old_pos = prev.nodes[label].pos
            new_pos = curr.nodes[label].pos
            delta   = _dist_3d(old_pos, new_pos)
            if delta > move_threshold_m:
                moved[label] = (old_pos, new_pos, delta)

        return SceneGraphDiff(
            appeared=appeared,
            disappeared=disappeared,
            moved=moved,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────

class LLMPromptBuilder:
    """
    Builds structured LLM prompts that combine the scene graph with
    the user's instruction and any known affordance weights.
    """

    @staticmethod
    def build_affordance_prompt(
        graph:            SceneGraph,
        user_instruction: str,
        known_weights:    dict[str, float],
        diff:             Optional[SceneGraphDiff] = None,
    ) -> str:
        """
        Generates a complete, structured LLM prompt for affordance assignment.

        The LLM receives:
        - Full scene graph with spatial relations
        - What changed since last frame (if available)
        - User instruction
        - Any explicitly known weights
        - Suggested default weights with reasoning
        - Clear output format specification
        """
        all_labels    = list(graph.nodes.keys())
        unmentioned   = [l for l in all_labels if l not in known_weights]
        suggested     = graph.suggest_default_weights()

        prompt_parts = []

        # system context
        prompt_parts.append(
            "You are the affordance reasoning module for a reactive robot "
            "manipulation system. Your job is to assign a numerical affordance "
            "weight to every object in the scene."
        )
        prompt_parts.append("")

        # scene graph
        prompt_parts.append(graph.to_prompt_string(include_weights_hint=False))

        # scene change
        if diff is not None and diff.is_significant():
            prompt_parts.append("── SCENE CHANGES SINCE LAST FRAME ──")
            prompt_parts.append(diff.to_natural_language())
            prompt_parts.append("")

        # instruction
        prompt_parts.append(f"User instruction: \"{user_instruction}\"")
        prompt_parts.append("")

        # known weights
        if known_weights:
            prompt_parts.append("── EXPLICITLY ASSIGNED WEIGHTS (do not change these) ──")
            for label, w in known_weights.items():
                sentiment = "GOAL (move toward)" if w > 0 else "OBSTACLE (avoid)"
                prompt_parts.append(f"  {label:20s}: {w:+.0f}  [{sentiment}]")
            prompt_parts.append("")

        # unmentioned objects with hints
        if unmentioned:
            prompt_parts.append(
                "── OBJECTS NOT MENTIONED BY USER (you must assign these) ──"
            )
            prompt_parts.append(
                "  Use the scene graph to reason about each one."
            )
            for label in unmentioned:
                node = graph.nodes[label]
                hint = suggested.get(label, -200)
                reasons = []
                if node.fragility > 0.6:
                    reasons.append(f"fragile ({node.fragility:.1f})")
                if label in graph.get_path_obstacles():
                    reasons.append("on path to goal")
                if not node.reachable:
                    reasons.append("out of reach")
                nearest = graph.get_nearest_to(label, k=2)
                if nearest:
                    nearest_str = ", ".join(
                        f"{l}({d:.2f}m)" for l, d in nearest
                    )
                    reasons.append(f"nearest: {nearest_str}")
                reason_str = " | ".join(reasons) if reasons else "no special flags"
                prompt_parts.append(
                    f"  {label:20s}  suggested={hint:+.0f}  "
                    f"(reason: {reason_str})"
                )
            prompt_parts.append("")

        # output format
        prompt_parts.append("── OUTPUT FORMAT ──")
        prompt_parts.append(
            "Return ONLY a valid JSON object with a weight for EVERY object."
        )
        prompt_parts.append("Weight scale:")
        prompt_parts.append("  +1000 = highest priority goal (move here)")
        prompt_parts.append("   +200 = secondary goal")
        prompt_parts.append("   -100 = low priority obstacle (can be relaxed)")
        prompt_parts.append("   -500 = moderate obstacle (avoid if possible)")
        prompt_parts.append("  -1000 = hard obstacle (never touch)")
        prompt_parts.append("")
        prompt_parts.append("Example output:")
        example = {l: known_weights.get(l, suggested.get(l, -200))
                   for l in all_labels}
        prompt_parts.append(json.dumps(example, indent=2))

        return "\n".join(prompt_parts)

    @staticmethod
    def build_scene_summary_prompt(graph: SceneGraph) -> str:
        """
        Simpler prompt: just ask the LLM to describe the scene in natural language.
        Useful for debugging and for grounding the system.
        """
        return (
            f"Describe this robot manipulation scene in 2-3 sentences, "
            f"focusing on what the robot can and cannot do:\n\n"
            f"{graph.to_prompt_string(include_weights_hint=False)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def _dist_2d(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _dist_3d(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _is_on_path(
    obj:   tuple,
    start: tuple,
    end:   tuple,
    tol:   float,
) -> bool:
    """
    True if obj is within tol metres of the line segment start→end (XY plane).
    Also checks that the object is between start and end (not beyond either end).
    """
    sx, sy = start[0], start[1]
    ex, ey = end[0],   end[1]
    ox, oy = obj[0],   obj[1]

    seg = math.sqrt((ex-sx)**2 + (ey-sy)**2)
    if seg < 1e-6:
        return False

    t = ((ox-sx)*(ex-sx) + (oy-sy)*(ey-sy)) / (seg**2)
    t = max(0.0, min(1.0, t))

    cx = sx + t*(ex-sx)
    cy = sy + t*(ey-sy)

    return math.sqrt((ox-cx)**2 + (oy-cy)**2) < tol


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def detections_from_perception_node(raw: list[dict]) -> list[Detection]:
    """
    Convert the dict format your perception_node.py already produces
    into Detection objects.

    raw = [
        {"label": "bottle", "pos": (0.30, 0.10, 0.15), "conf": 0.85,
         "size": (0.08, 0.30, 0.08)},
        ...
    ]
    """
    return [
        Detection(
            label=d["label"],
            pos=tuple(d["pos"]),
            conf=d.get("conf", 1.0),
            bbox_px=tuple(d.get("bbox_px", (0, 0, 0, 0))),
            size_est=tuple(d.get("size", (0.0, 0.0, 0.0))),
        )
        for d in raw
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    builder = SceneGraphBuilder(
        robot_pos  = (-0.5, 0.0, 0.0),
        camera_pos = (0.5, -1.4, 0.825),
    )

    # Frame 1 — initial scene
    raw_detections_frame1 = [
        {"label": "bottle",      "pos": (0.30, 0.10, 0.15), "conf": 0.85},
        {"label": "vase",        "pos": (-0.14, 0.26, 0.10), "conf": 0.72},
        {"label": "cup",         "pos": (0.10, 0.22, 0.09), "conf": 0.52},
        {"label": "sports ball", "pos": (0.55, 0.30, 0.08), "conf": 0.79},
        {"label": "boot",        "pos": (0.42, 0.10, 0.12), "conf": 0.07},
    ]

    graph1, diff1 = builder.build(
        detections=[Detection(**{
            "label": d["label"],
            "pos": tuple(d["pos"]),
            "conf": d["conf"],
        }) for d in raw_detections_frame1],
        goal_label="sports ball",
    )

    print(graph1.to_prompt_string())

    visualize(graph1, save_path="/tmp/scene_graph_viz.png")
    

    # Frame 2 — someone moved the boot
    raw_detections_frame2 = [
        {"label": "bottle",      "pos": (0.30, 0.10, 0.15), "conf": 0.85},
        {"label": "vase",        "pos": (-0.14, 0.26, 0.10), "conf": 0.72},
        {"label": "cup",         "pos": (0.10, 0.22, 0.09), "conf": 0.52},
        {"label": "sports ball", "pos": (0.55, 0.30, 0.08), "conf": 0.79},
        {"label": "boot",        "pos": (0.15, 0.18, 0.12), "conf": 0.08},  # moved!
    ]

    graph2, diff2 = builder.build(
        detections=[Detection(**{
            "label": d["label"],
            "pos": tuple(d["pos"]),
            "conf": d["conf"],
        }) for d in raw_detections_frame2],
        goal_label="sports ball",
    )

    print("=== SCENE DIFF (boot was moved) ===")
    print(diff2.to_natural_language())
    print(f"Significant change: {diff2.is_significant()}")
    print()

    # Query API examples
    print("=== QUERY API ===")
    print(f"Path obstacles:      {graph2.get_path_obstacles()}")
    print(f"Fragile objects:     {graph2.get_fragile_objects(0.5)}")
    print(f"Reachable objects:   {graph2.get_reachable_objects()}")
    print(f"Nearest to ball:     {graph2.get_nearest_to('sports ball', 3)}")
    print(f"Clusters:            {graph2.get_cluster_summary()}")
    print(f"Objects left of cup: {graph2.get_objects_with_relation('cup', 'left_of')}")
    print()

    # LLM prompt with user-mentioned weights + unmentioned objects
    known_weights = {
        "sports ball": +200,
        "bottle":      -1000,
    }

    prompt = LLMPromptBuilder.build_affordance_prompt(
        graph=graph2,
        user_instruction="pick up the red ball, avoid the bottle",
        known_weights=known_weights,
        diff=diff2,
    )

    print("=== LLM PROMPT (first 60 lines) ===")
    for line in prompt.split("\n")[:60]:
        print(line)
    print("...")

    # JSON export
    print()
    print("=== JSON EXPORT (first 20 lines) ===")
    json_str = graph2.to_json()
    for line in json_str.split("\n")[:20]:
        print(line)
    print("...")