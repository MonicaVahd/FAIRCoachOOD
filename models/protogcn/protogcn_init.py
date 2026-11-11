# ----------------------------------------------------------------------------
# Third-party code: ProtoGCN.  Upstream: https://github.com/firework8/ProtoGCN (protogcn/models/gcns/utils/init_func.py)
# Compared with upstream commit ddf7f274f9f5 (2025-08-29); the commit the file was originally taken from is not recorded.
# Copyright (c) 2024 Hongda Liu
# Licence: MIT. Full text: THIRD_PARTY_LICENSES/protogcn_LICENSE.txt
# Modification notice: No change to the code lines was found: all non-comment, non-blank lines are
#   identical to the upstream file (compared 2026-11-10 against the commit above). Comments or
#   blank lines may differ.
# ----------------------------------------------------------------------------
import math
import torch.nn as nn


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
