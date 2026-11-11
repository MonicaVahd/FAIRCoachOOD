# ----------------------------------------------------------------------------
# Third-party code: HD-GCN.  Upstream: https://github.com/Jho-Yonsei/HD-GCN (graph/ntu_rgb_d_hierarchy.py)
# Compared with upstream commit 672da2c3d84c (2023-10-05); the commit the file was originally taken from is not recorded.
# Copyright (c) 2023 Jungho Lee
# Licence: MIT. Full text: THIRD_PARTY_LICENSES/hd-gcn_LICENSE.txt
# Modification notice: MODIFIED. This file differs from the upstream file (line similarity 0.596 to
#   graph/ntu_rgb_d_hierarchy.py, comments and blank lines ignored, compared 2026-11-10 against the
#   commit above). A record of the individual changes was not kept when the file was adapted. No
#   upstream file matches closely; this file adapts HD-GCN's hierarchical graph definition to the
#   OpenPose-18 layout.
# ----------------------------------------------------------------------------
import numpy as np

from net import hdgcn_graph_tools as tools

num_node = 25


class Graph:
    def __init__(self, CoM=21, labeling_mode='spatial'):
        self.num_node = num_node
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_hierarchical_graph(
                num_node, tools.get_edgeset(dataset='NTU', CoM=self.CoM)
            )
        else:
            raise ValueError()
        return A, self.CoM
