#!/bin/bash
# Prepare ASL dataset for Kaggle upload

set -e

echo "=========================================="
echo "Preparing ASL Dataset for Kaggle"
echo "=========================================="

# Create directory structure
echo -e "\n[1/4] Creating directory structure..."
mkdir -p kaggle_dataset/landmarks

# Copy data
echo -e "\n[2/4] Copying landmark data..."
for letter in {A..Z}; do
    echo "  Copying class $letter..."
    cp -r ../data/raw_gestures/$letter kaggle_dataset/landmarks/
    count=$(find kaggle_dataset/landmarks/$letter -name "*.npy" | wc -l)
    echo "    ✓ $count samples"
done

# Copy metadata
echo -e "\n[3/4] Copying metadata files..."
cp dataset-metadata.json kaggle_dataset/
cp README.md kaggle_dataset/
cp example-usage.py kaggle_dataset/

# Create statistics file
echo -e "\n[4/4] Generating statistics..."
cat > kaggle_dataset/STATISTICS.txt << EOF
ASL Alphabet Hand Landmarks Dataset - Statistics
================================================

Generated: $(date)

Class Distribution:
EOF

for letter in {A..Z}; do
    count=$(find kaggle_dataset/landmarks/$letter -name "*.npy" | wc -l)
    printf "  %s: %4d samples\n" "$letter" "$count" >> kaggle_dataset/STATISTICS.txt
done

total=$(find kaggle_dataset/landmarks -name "*.npy" | wc -l)
echo "" >> kaggle_dataset/STATISTICS.txt
echo "Total: $total samples" >> kaggle_dataset/STATISTICS.txt
echo "Classes: 26" >> kaggle_dataset/STATISTICS.txt

# Show size
size=$(du -sh kaggle_dataset | cut -f1)
echo -e "\n✓ Dataset prepared!"
echo -e "\nDataset size: $size"
echo "Location: ./kaggle_dataset/"

# Show structure
echo -e "\nDirectory structure:"
tree -L 2 kaggle_dataset/ 2>/dev/null || find kaggle_dataset -type d | head -30

echo -e "\n=========================================="
echo "Next steps:"
echo "=========================================="
echo ""
echo "Option A - Upload via Kaggle CLI:"
echo "  1. Install Kaggle CLI: pip install kaggle"
echo "  2. Setup API token: https://kaggle.com/settings"
echo "  3. Run: cd kaggle_dataset && kaggle datasets create -p ."
echo ""
echo "Option B - Upload via Web Interface:"
echo "  1. Compress: cd kaggle_dataset && zip -r ../asl-landmarks.zip ."
echo "  2. Go to: https://kaggle.com/datasets"
echo "  3. Click 'New Dataset' and upload zip"
echo ""
echo "=========================================="

