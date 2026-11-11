"""
models/builders.py — Factory functions for all nine benchmark models.

Included models (licence: MIT, Apache-2.0, BSD 2-clause):
  build_stgcn_model, build_blockgcn_model, build_hdgcn_model,
  build_protogcn_model, build_skateformer_model, build_hyperformer_model

Excluded models (licence issues — download via scripts/fetch_upstream.py):
  build_ctrgcn_model, build_infogcn_model, build_freqmixformer_model

Usage
-----
  import sys; sys.path.insert(0, '.')  # from repo root
  from models.builders import build_stgcn_model
  model = build_stgcn_model(device='cuda')
"""

import os
import sys
import types

import torch

# The model files were written for a package called "net" (the original development tree kept them in
# lib/net/ and they import each other as `net.<module>` / `net.utils.<module>`).  Here they live in one
# folder per model, so a "net" package whose search path is those folders is registered.
_HERE = os.path.dirname(os.path.abspath(__file__))
_NET_DIRS = [os.path.join(_HERE, d) for d in
             ("st_gcn", "hdgcn", "protogcn", "skateformer", "blockgcn", "hyperformer")]
if "net" not in sys.modules:
    _net = types.ModuleType("net")
    _net.__path__ = _NET_DIRS
    sys.modules["net"] = _net

# ── ST-GCN (BSD 2-clause) ────────────────────────────────────────────────────

def build_stgcn_model(checkpoint_path=None, device="cpu"):
    """Return ST-GCN in eval mode (paper §3.2, Table 2 row 1)."""
    from net.st_gcn import Model

    model = Model(
        in_channels=3,
        num_class=2,
        graph_args={"layout": "openpose", "strategy": "spatial"},
        edge_importance_weighting=True,
        dropout=0.5,
    ).to(device)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()


# ── BlockGCN (Apache-2.0, modified) ─────────────────────────────────────────

def build_blockgcn_model(checkpoint_path=None, device="cpu"):
    """Return BlockGCN in eval mode (topology encoder replaced, paper Table 2)."""
    from net.blockgcn import Model as BlockGCNModel
    import net.utils.graph  # noqa: F401 -- registers the submodule for import_class()

    model = BlockGCNModel(
        num_class=2,
        num_point=18,
        num_person=1,
        in_channels=3,
        graph="net.utils.graph.Graph",
        graph_args={"layout": "openpose", "strategy": "spatial"},
        adaptive=True,
        drop_out=0,
    ).to(device)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()


# ── HD-GCN (MIT) ─────────────────────────────────────────────────────────────

def build_hdgcn_model(checkpoint_path=None, device="cpu"):
    """Return HD-GCN with OpenPose-18 graph, centre joint 8 (RHip)."""
    from net.hdgcn import Model as HDGCNModel
    import net.hdgcn_graph_openpose  # noqa: F401

    model = HDGCNModel(
        num_class=2,
        num_point=18,
        num_person=1,
        in_channels=3,
        graph="net.hdgcn_graph_openpose.Graph",
        graph_args={"labeling_mode": "spatial", "CoM": 8},
        adaptive=True,
        drop_out=0,
    ).to(device)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()


# ── ProtoGCN (MIT) ───────────────────────────────────────────────────────────

def build_protogcn_model(checkpoint_path=None, device="cpu"):
    """Return ProtoGCN with base_channels=96 (paper Table 2)."""
    from net.protogcn import ProtoGCNClassifier

    model = ProtoGCNClassifier(
        num_class=2,
        graph_cfg={"layout": "openpose", "mode": "spatial"},
        in_channels=3,
        base_channels=96,
        num_person=1,
        data_bn_type="VC",
    ).to(device)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()


# ── SkateFormer (MIT) ────────────────────────────────────────────────────────

def build_skateformer_model(checkpoint_path=None, device="cpu"):
    """Return SkateFormer with index_t=False, num_frames=100, 18 joints."""
    from net.skateformer import SkateFormer

    model = SkateFormer(
        in_channels=3,
        num_classes=2,
        num_people=1,
        num_frames=100,
        num_points=18,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        channels=(96, 192, 192, 192),
        kernel_size=7,
        num_heads=32,
        type_1_size=(1, 1),
        type_2_size=(1, 1),
        type_3_size=(1, 1),
        type_4_size=(1, 1),
        index_t=False,
    ).to(device)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()


# ── HyperFormer (Apache-2.0, modified) ───────────────────────────────────────

def build_hyperformer_model(checkpoint_path=None, device="cpu"):
    """Return HyperFormer with OpenPose-18 graph."""
    from net.hyperformer import Model as HyperFormerModel
    from net.hyperformer_graph import JOINT_LABEL
    import net.hyperformer_graph  # noqa: F401

    model = HyperFormerModel(
        num_class=2,
        num_point=18,
        num_person=1,
        in_channels=3,
        graph="net.hyperformer_graph.Graph",
        graph_args={"labeling_mode": "spatial"},
        joint_label=JOINT_LABEL,
    ).to(device)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()


# ── Excluded models (fetch with scripts/fetch_upstream.py) ───────────────────

def _excluded(name):
    raise ImportError(
        f"{name} is not included in this release because its upstream "
        "repository has no compatible licence.  Run:\n"
        "  python scripts/fetch_upstream.py\n"
        "to download it from its original repository."
    )


def build_ctrgcn_model(checkpoint_path=None, device="cpu"):      # noqa: D401
    """NOT INCLUDED — CC BY-NC 4.0.  Use scripts/fetch_upstream.py."""
    _excluded("CTR-GCN")


def build_infogcn_model(checkpoint_path=None, device="cpu"):     # noqa: D401
    """NOT INCLUDED — no licence file.  Use scripts/fetch_upstream.py."""
    _excluded("InfoGCN")


def build_freqmixformer_model(checkpoint_path=None, device="cpu"):  # noqa: D401
    """NOT INCLUDED — no licence file; substantially adapted port."""
    _excluded("FreqMixFormer")


# ── Registry ─────────────────────────────────────────────────────────────────

BUILDERS = {
    "ST-GCN":        build_stgcn_model,
    "BlockGCN":      build_blockgcn_model,
    "HD-GCN":        build_hdgcn_model,
    "ProtoGCN":      build_protogcn_model,
    "SkateFormer":   build_skateformer_model,
    "HyperFormer":   build_hyperformer_model,
    # --- fetch first ---
    "CTR-GCN":       build_ctrgcn_model,
    "InfoGCN":       build_infogcn_model,
    "FreqMixFormer": build_freqmixformer_model,
}
