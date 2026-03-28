# =============================================================================
# dataset_analysis.py  ─  Visualise class distribution in your dataset
# Run:  python dataset_analysis.py
# =============================================================================

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR = "final_dataset/train"
TEST_DIR  = "final_dataset/test"

# ── Check folder exists ───────────────────────────────────────────────────────
if not os.path.exists(TRAIN_DIR):
    print(f"❌  Dataset folder not found: {TRAIN_DIR}")
    print("    Make sure your dataset is at  final_dataset/train/")
    exit(1)

# ── Count images per class ────────────────────────────────────────────────────
def count_images(directory):
    counts = {}
    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            # count only image files
            imgs = [f for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
            counts[class_name] = len(imgs)
    return counts

train_counts = count_images(TRAIN_DIR)
test_counts  = count_images(TEST_DIR) if os.path.exists(TEST_DIR) else {}

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n📊  Dataset Class Distribution")
print("=" * 40)
print(f"{'Class':<15} {'Train':>8} {'Test':>8} {'Total':>8}")
print("-" * 40)

total_train = total_test = 0
for cls in sorted(train_counts.keys()):
    tr  = train_counts.get(cls, 0)
    te  = test_counts.get(cls, 0)
    tot = tr + te
    total_train += tr
    total_test  += te
    print(f"{cls:<15} {tr:>8} {te:>8} {tot:>8}")

print("-" * 40)
print(f"{'TOTAL':<15} {total_train:>8} {total_test:>8} {total_train+total_test:>8}")
print()

# ── Plot ──────────────────────────────────────────────────────────────────────
classes  = sorted(train_counts.keys())
tr_vals  = [train_counts.get(c, 0) for c in classes]
te_vals  = [test_counts.get(c, 0)  for c in classes]

x      = range(len(classes))
width  = 0.4

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar([i - width/2 for i in x], tr_vals, width,
               label="Train", color="#1565C0", alpha=0.85)
bars2 = ax.bar([i + width/2 for i in x], te_vals, width,
               label="Test",  color="#4CAF50", alpha=0.85)

# Value labels on top of each bar
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)

ax.set_xticks(list(x))
ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=12)
ax.set_xlabel("Waste Category", fontsize=13)
ax.set_ylabel("Number of Images", fontsize=13)
ax.set_title("Dataset Class Distribution  (Train vs Test)", fontsize=15, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# Save and show
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/dataset_distribution.png", dpi=150)
print("📁  Chart saved to  assets/dataset_distribution.png")
plt.show()