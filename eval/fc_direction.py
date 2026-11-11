"""Break down 141 FC cells by collapse direction (all-incorrect vs all-correct)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collapse import (
    CANONICAL_MODELS,
    parse_confusion_file,
    CONFUSION_PATH,
)

cells = parse_confusion_file(CONFUSION_PATH)
fc = [c for c in cells if c.collapse == "FC"]
n = len(fc)

all_incorrect = [c for c in fc if c.pred_correct == 0]
all_correct = [c for c in fc if c.pred_incorrect == 0]

print(f"Total FC cells: {n}")
print(
    f"Predict ALL incorrect (pred_correct=0): "
    f"{len(all_incorrect)} ({100 * len(all_incorrect) / n:.6f}%)"
)
print(
    f"Predict ALL correct   (pred_incorrect=0): "
    f"{len(all_correct)} ({100 * len(all_correct) / n:.6f}%)"
)
print()

print("Per-model (FC cells only):")
print(f"{'Model':<14} {'FC':>4} {'AllInc':>7} {'AllCor':>7} {'Inc%':>8}")
for model in CANONICAL_MODELS:
    sub = [c for c in fc if c.model == model]
    inc = sum(1 for c in sub if c.pred_correct == 0)
    cor = sum(1 for c in sub if c.pred_incorrect == 0)
    pct = 100 * inc / len(sub) if sub else 0
    print(f"{model:<14} {len(sub):>4} {inc:>7} {cor:>7} {pct:>7.2f}%")

print()
print("All-correct FC cells (pred_incorrect=0):")
for c in sorted(all_correct, key=lambda x: (x.model, x.condition)):
    print(f"  {c.model:<14} {c.condition}  pred_correct={c.pred_correct}  n={c.tp+c.tn+c.fp+c.fn}")

bad = [c for c in fc if c.pred_correct > 0 and c.pred_incorrect > 0]
print(f"\nSanity: FC with both pred classes > 0: {len(bad)}")
