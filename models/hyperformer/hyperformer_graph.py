# ----------------------------------------------------------------------------
# Third-party code: HyperFormer.  Upstream: https://github.com/ZhouYuxuanYX/Hyperformer (graph/ucla.py)
# Compared with upstream commit c35d996bcf51 (2023-10-07); the commit the file was originally taken from is not recorded.
# Copyright 2022 Yuxuan Zhou
# Licence: Apache-2.0. Full text: THIRD_PARTY_LICENSES/hyperformer_LICENSE.txt
# Modification notice: MODIFIED. This file differs from the upstream file (line similarity 0.222 to
#   graph/ucla.py, comments and blank lines ignored, compared 2026-11-10 against the commit above).
#   A record of the individual changes was not kept when the file was adapted. No upstream file
#   matches closely; this file defines the 5-group OpenPose-18 body-part labelling used for
#   hyperedges (paper Table 2).
# ----------------------------------------------------------------------------
"""HyperFormer graph for OpenPose-18 skeleton layout."""

from net.utils.graph import Graph as BaseGraph

NUM_NODE = 18

# Joint groups (0-indexed OpenPose-18):
# Group 0 (head):       0, 1, 14, 15, 16, 17
# Group 1 (right arm):  2, 3, 4
# Group 2 (left arm):   5, 6, 7
# Group 3 (right leg):  8, 9, 10
# Group 4 (left leg):   11, 12, 13
JOINT_LABEL = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0, 0
]

JOINT_GROUPS = [
    [0, 1, 14, 15, 16, 17],
    [2, 3, 4],
    [5, 6, 7],
    [8, 9, 10],
    [11, 12, 13],
]


class Graph:
    def __init__(self, labeling_mode="spatial"):
        self.num_node = NUM_NODE
        self.joint_label = JOINT_LABEL
        self.joint_groups = JOINT_GROUPS
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            return BaseGraph(layout="openpose", strategy="spatial").A
        raise ValueError()
