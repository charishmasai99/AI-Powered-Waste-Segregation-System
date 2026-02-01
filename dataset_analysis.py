import os
import matplotlib.pyplot as plt

TRAIN_DIR = "final_dataset/train"

class_counts = {}

# Count images per class
for class_name in os.listdir(TRAIN_DIR):
    class_path = os.path.join(TRAIN_DIR, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))

print("📊 Dataset Class Distribution:\n")
for cls, count in class_counts.items():
    print(f"{cls} : {count}")

# Plot bar chart
plt.figure(figsize=(8, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel("Waste Category")
plt.ylabel("Number of Images")
plt.title("Training Dataset Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
