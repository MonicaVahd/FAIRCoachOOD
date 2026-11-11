# UI-PRMD rerun: Table 6 value vs rerun macro F1 (117 cells)

Companion table to `KNOWN_DIFFERENCES.md`, entry D4. "Table 6" values are the values in `results/ood_results.csv`
(they equal the paper's Table 6). "Rerun" values are computed from the confusion counts in
`results/uiprmd_rerun/uiprmd_rerun_confusion.txt`.

Conditions: B7-B9, C4, C5, C7-C10 (Category A, retrained, 81 cells); B10-B12, C6 (Category B, existing checkpoints re-evaluated, 36 cells).

Rule: the three-way rule of D4. Status from the confusion counts: FC if one class receives no prediction; PC if not FC and one class receives
5 or fewer; OK otherwise. recovery = FC or PC → OK; regression = OK → FC or PC; FC↔PC = FC → PC or PC → FC.
Counts in this table: 11 recoveries, 8 regressions, 6 FC↔PC (25), the same as D4.

| Model | Cond | Cat | Table 6 MF1 | Table 6 status | Rerun MF1 | Rerun status | Δ | Change |
|-------|------|-----|------------|---------|-----------|-------|---|--------|
| ST-GCN | B7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| CTR-GCN | B7 | A | 41.5 | FC | 22.5 | FC | -19.0 | same |
| InfoGCN | B7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| BlockGCN | B7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| HD-GCN | B7 | A | 23.6 | PC | 41.5 | FC | +17.9 | **FC↔PC** |
| ProtoGCN | B7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| SkateFormer | B7 | A | 41.0 | PC | 22.5 | FC | -18.5 | **FC↔PC** |
| HyperFormer | B7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| FreqMixFormer | B7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| ST-GCN | B8 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| CTR-GCN | B8 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| InfoGCN | B8 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| BlockGCN | B8 | A | 32.8 | OK | 24.2 | FC | -8.6 | **regression** |
| HD-GCN | B8 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| ProtoGCN | B8 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| SkateFormer | B8 | A | 39.6 | PC | 58.0 | OK | +18.5 | **recovery** |
| HyperFormer | B8 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| FreqMixFormer | B8 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| ST-GCN | B9 | A | 29.9 | FC | 29.9 | FC | +0.0 | same |
| CTR-GCN | B9 | A | 29.9 | FC | 29.9 | FC | +0.0 | same |
| InfoGCN | B9 | A | 29.9 | FC | 29.9 | FC | +0.0 | same |
| BlockGCN | B9 | A | 32.2 | OK | 41.9 | OK | +9.7 | same |
| HD-GCN | B9 | A | 29.9 | FC | 29.9 | FC | +0.0 | same |
| ProtoGCN | B9 | A | 29.9 | FC | 29.9 | FC | +0.0 | same |
| SkateFormer | B9 | A | 35.2 | OK | 36.5 | FC | +1.2 | **regression** |
| HyperFormer | B9 | A | 29.9 | FC | 29.9 | FC | +0.0 | same |
| FreqMixFormer | B9 | A | 36.5 | FC | 29.9 | FC | -6.6 | same |
| ST-GCN | C4 | A | 46.4 | OK | 49.6 | OK | +3.2 | same |
| CTR-GCN | C4 | A | 54.4 | OK | 57.0 | OK | +2.6 | same |
| InfoGCN | C4 | A | 52.4 | OK | 52.3 | OK | -0.1 | same |
| BlockGCN | C4 | A | 36.5 | FC | 37.1 | PC | +0.6 | **FC↔PC** |
| HD-GCN | C4 | A | 52.7 | OK | 27.9 | OK | -24.9 | same |
| ProtoGCN | C4 | A | 29.9 | FC | 45.5 | OK | +15.6 | **recovery** |
| SkateFormer | C4 | A | 29.9 | FC | 45.3 | OK | +15.5 | **recovery** |
| HyperFormer | C4 | A | 48.5 | OK | 28.2 | OK | -20.3 | same |
| FreqMixFormer | C4 | A | 29.3 | PC | 31.3 | OK | +1.9 | **recovery** |
| ST-GCN | C5 | A | 37.3 | OK | 44.6 | OK | +7.3 | same |
| CTR-GCN | C5 | A | 29.9 | FC | 51.4 | OK | +21.5 | **recovery** |
| InfoGCN | C5 | A | 47.3 | OK | 50.7 | OK | +3.3 | same |
| BlockGCN | C5 | A | 36.5 | FC | 36.5 | FC | +0.0 | same |
| HD-GCN | C5 | A | 29.9 | FC | 29.9 | FC | +0.0 | same |
| ProtoGCN | C5 | A | 29.9 | FC | 28.4 | OK | -1.4 | **recovery** |
| SkateFormer | C5 | A | 36.5 | FC | 36.5 | FC | +0.0 | same |
| HyperFormer | C5 | A | 29.9 | FC | 29.9 | FC | +0.0 | same |
| FreqMixFormer | C5 | A | 36.0 | OK | 38.5 | OK | +2.5 | same |
| ST-GCN | C7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| CTR-GCN | C7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| InfoGCN | C7 | A | 22.5 | FC | 41.5 | FC | +19.0 | same |
| BlockGCN | C7 | A | 27.2 | OK | 41.5 | FC | +14.3 | **regression** |
| HD-GCN | C7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| ProtoGCN | C7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| SkateFormer | C7 | A | 22.5 | FC | 41.2 | PC | +18.7 | **FC↔PC** |
| HyperFormer | C7 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| FreqMixFormer | C7 | A | 41.5 | FC | 22.5 | FC | -19.0 | same |
| ST-GCN | C8 | A | 56.6 | OK | 29.9 | FC | -26.7 | **regression** |
| CTR-GCN | C8 | A | 29.9 | FC | 47.6 | OK | +17.7 | **recovery** |
| InfoGCN | C8 | A | 29.6 | OK | 49.2 | OK | +19.6 | same |
| BlockGCN | C8 | A | 45.3 | OK | 36.5 | FC | -8.8 | **regression** |
| HD-GCN | C8 | A | 28.8 | OK | 31.9 | OK | +3.1 | same |
| ProtoGCN | C8 | A | 36.5 | OK | 28.9 | OK | -7.6 | same |
| SkateFormer | C8 | A | 36.5 | FC | 36.5 | FC | +0.0 | same |
| HyperFormer | C8 | A | 29.3 | PC | 50.3 | OK | +21.0 | **recovery** |
| FreqMixFormer | C8 | A | 29.8 | PC | 34.3 | OK | +4.5 | **recovery** |
| ST-GCN | C9 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| CTR-GCN | C9 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| InfoGCN | C9 | A | 52.1 | OK | 41.9 | OK | -10.2 | same |
| BlockGCN | C9 | A | 30.8 | OK | 50.6 | OK | +19.8 | same |
| HD-GCN | C9 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| ProtoGCN | C9 | A | 24.2 | FC | 26.4 | OK | +2.2 | **recovery** |
| SkateFormer | C9 | A | 48.7 | OK | 24.2 | FC | -24.5 | **regression** |
| HyperFormer | C9 | A | 23.6 | PC | 24.2 | FC | +0.6 | **FC↔PC** |
| FreqMixFormer | C9 | A | 24.2 | FC | 24.2 | FC | +0.0 | same |
| ST-GCN | C10 | A | 41.5 | FC | 22.5 | FC | -19.0 | same |
| CTR-GCN | C10 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| InfoGCN | C10 | A | 22.5 | FC | 22.5 | FC | +0.0 | same |
| BlockGCN | C10 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| HD-GCN | C10 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| ProtoGCN | C10 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| SkateFormer | C10 | A | 41.0 | PC | 56.5 | OK | +15.5 | **recovery** |
| HyperFormer | C10 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| FreqMixFormer | C10 | A | 41.5 | FC | 41.5 | FC | +0.0 | same |
| ST-GCN | B10 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| CTR-GCN | B10 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| InfoGCN | B10 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| BlockGCN | B10 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| HD-GCN | B10 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| ProtoGCN | B10 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| SkateFormer | B10 | B | 41.7 | OK | 33.3 | FC | -8.3 | **regression** |
| HyperFormer | B10 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| FreqMixFormer | B10 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| ST-GCN | B11 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| CTR-GCN | B11 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| InfoGCN | B11 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| BlockGCN | B11 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| HD-GCN | B11 | B | 44.5 | OK | 35.5 | PC | -9.0 | **regression** |
| ProtoGCN | B11 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| SkateFormer | B11 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| HyperFormer | B11 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| FreqMixFormer | B11 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| ST-GCN | B12 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| CTR-GCN | B12 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| InfoGCN | B12 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| BlockGCN | B12 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| HD-GCN | B12 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| ProtoGCN | B12 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| SkateFormer | B12 | B | 34.4 | PC | 33.3 | FC | -1.1 | **FC↔PC** |
| HyperFormer | B12 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| FreqMixFormer | B12 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| ST-GCN | C6 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| CTR-GCN | C6 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| InfoGCN | C6 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| BlockGCN | C6 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| HD-GCN | C6 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| ProtoGCN | C6 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| SkateFormer | C6 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
| HyperFormer | C6 | B | 32.0 | OK | 67.5 | OK | +35.6 | same |
| FreqMixFormer | C6 | B | 33.3 | FC | 33.3 | FC | +0.0 | same |
