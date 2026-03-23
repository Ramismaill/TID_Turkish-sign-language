"""
graph.py — ST-GCN Adjacency Matrix for MediaPipe Holistic (56 nodes)

Node mapping (original MediaPipe → new 56-node space):
  Pose nodes 11-24  → new indices  0-13   (14 nodes, upper body only)
  LH   nodes  0-20  → new indices 14-34   (21 nodes)
  RH   nodes  0-20  → new indices 35-55   (21 nodes)

Pruned: face (pose 0-10) and legs (pose 25-32)

New pose index reference:
   0 = left shoulder  (orig 11)    1 = right shoulder (orig 12)
   2 = left elbow     (orig 13)    3 = right elbow    (orig 14)
   4 = left wrist     (orig 15)    5 = right wrist    (orig 16)
   6 = left pinky     (orig 17)    7 = right pinky    (orig 18)
   8 = left index     (orig 19)    9 = right index    (orig 20)
  10 = left thumb     (orig 21)   11 = right thumb    (orig 22)
  12 = left hip       (orig 23)   13 = right hip      (orig 24)
"""

import numpy as np
from collections import deque

NUM_NODES = 56

# ── Feature slice indices from raw (64, 225) array ──────────────────────────
KEEP_INDICES = list(range(33, 75)) + list(range(99, 162)) + list(range(162, 225))
assert len(KEEP_INDICES) == NUM_NODES * 3, f"Expected {NUM_NODES*3}, got {len(KEEP_INDICES)}"

# ── Bone pairs for bone stream computation ───────────────────────────────────
# Each tuple: (child_node, parent_node) in 56-node space
BONE_PAIRS = [
    # Pose skeleton (0-13)
    (0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 3),
    (6, 4), (7, 5), (8, 4), (9, 5), (10, 4), (11, 5),
    (12, 0), (13, 1),
    # Left hand (14-34)
    (14, 4),
    (15, 14), (16, 15), (17, 16), (18, 17),
    (19, 14), (20, 19), (21, 20), (22, 21),
    (23, 14), (24, 23), (25, 24), (26, 25),
    (27, 14), (28, 27), (29, 28), (30, 29),
    (31, 14), (32, 31), (33, 32), (34, 33),
    # Right hand (35-55)
    (35, 5),
    (36, 35), (37, 36), (38, 37), (39, 38),
    (40, 35), (41, 40), (42, 41), (43, 42),
    (44, 35), (45, 44), (46, 45), (47, 46),
    (48, 35), (49, 48), (50, 49), (51, 50),
    (52, 35), (53, 52), (54, 53), (55, 54),
]


def _hand_edges(offset: int):
    edges = []
    for fb in [1, 5, 9, 13, 17]:
        edges.append((offset, offset + fb))
    for chain_start in [1, 5, 9, 13, 17]:
        for i in range(3):
            edges.append((offset + chain_start + i,
                          offset + chain_start + i + 1))
    return edges


def _get_all_edges():
    pose_edges = [
        (0,  1),
        (0,  2), (2,  4),
        (1,  3), (3,  5),
        (0, 12), (1, 13),
        (12, 13),
        (4,  6), (4,  8), (4, 10),
        (5,  7), (5,  9), (5, 11),
        (4,  5),
        (2,  3),
    ]
    bridge_edges = [
        (4, 14),
        (5, 35),
    ]
    lh_edges = _hand_edges(14)
    rh_edges = _hand_edges(35)
    return pose_edges + bridge_edges + lh_edges + rh_edges


def build_adjacency() -> np.ndarray:
    N = NUM_NODES
    edges = _get_all_edges()

    A_sym = np.zeros((N, N), dtype=np.float32)
    for i, j in edges:
        A_sym[i, j] = 1.0
        A_sym[j, i] = 1.0

    dist = [-1] * N
    dist[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in range(N):
            if A_sym[u, v] > 0 and dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    max_d = max(d for d in dist if d >= 0)
    dist = [d if d >= 0 else max_d + 1 for d in dist]

    A_self = np.eye(N, dtype=np.float32)
    A_in   = np.zeros((N, N), dtype=np.float32)
    A_out  = np.zeros((N, N), dtype=np.float32)

    for i, j in edges:
        if dist[i] > dist[j]:
            A_in[i, j]  = 1.0
            A_out[j, i] = 1.0
        elif dist[i] < dist[j]:
            A_out[i, j] = 1.0
            A_in[j, i]  = 1.0
        else:
            A_in[i, j]  = 1.0
            A_in[j, i]  = 1.0

    def _row_normalize(a):
        row_sum = a.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return a / row_sum

    A_self = _row_normalize(A_self)
    A_in   = _row_normalize(A_in)
    A_out  = _row_normalize(A_out)

    return np.stack([A_self, A_in, A_out], axis=0)


ADJACENCY = build_adjacency()