# Error Analysis Module

Comprehensive error analysis tools for CIFAR-100 image captioning.

## Features

### 1. **Confusion Matrix Analysis**
- Visualizes confusion between all 100 CIFAR-100 classes
- Highlights the top 20 most confused classes
- Heatmap showing prediction patterns

### 2. **Per-Class Performance**
- Accuracy for each of the 100 classes
- Identifies best and worst performing classes
- Detailed statistics per class

### 3. **Superclass Analysis**
- Performance grouped by 20 CIFAR-100 superclasses
- Identifies which categories (e.g., animals, vehicles) perform best/worst
- Average accuracy across related classes

### 4. **Confused Class Pairs**
- Top 20 most frequently confused class pairs
- Helps identify systematic errors (e.g., "leopard" → "tiger")

### 5. **Sample Visualizations**
- Top 10 successful predictions with images
- Top 10 failed predictions with images
- Visual comparison of predictions vs ground truth

## Usage

### Basic Analysis
```bash
python analysis/error_analysis.py \
    --ckpt runs_continued/best.pt \
    --batch_size 256
```

### Custom Output Directory
```bash
python analysis/error_analysis.py \
    --ckpt runs_continued/best.pt \
    --batch_size 256 \
    --output_dir my_analysis_results
```

### All Options
```bash
python analysis/error_analysis.py \
    --ckpt runs_continued/best.pt \
    --batch_size 256 \
    --max_len 8 \
    --output_dir outputs/analysis \
    --data_root ./data
```

## Output Files

After running, you'll find in `outputs/analysis/`:

1. **confusion_matrix.png** - Visual confusion matrix heatmap
2. **sample_successes.png** - Grid of 10 successful predictions
3. **sample_failures.png** - Grid of 10 failed predictions
4. **analysis_report.txt** - Comprehensive text report with:
   - Overall accuracy
   - Top 10 best/worst classes
   - Most confused pairs
   - Superclass performance breakdown
5. **analysis_results.json** - Machine-readable detailed results

## Requirements

Install additional dependencies:
```bash
pip install matplotlib seaborn
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Understanding the Results

### Confusion Matrix
- **Diagonal**: Correct predictions (higher is better)
- **Off-diagonal**: Confusions (lower is better)
- Dark red indicates many confusions

### Superclass Performance
CIFAR-100 has 20 superclasses:
- `aquatic_mammals`, `fish`, `flowers`, `food_containers`
- `fruit_and_vegetables`, `household_electrical_devices`
- `household_furniture`, `insects`, `large_carnivores`
- `large_man-made_outdoor_things`, `large_natural_outdoor_scenes`
- `large_omnivores_and_herbivores`, `medium_mammals`
- `non-insect_invertebrates`, `people`, `reptiles`
- `small_mammals`, `trees`, `vehicles_1`, `vehicles_2`

### Common Issues to Look For

1. **Similar-looking classes confused** (e.g., leopard ↔ tiger)
2. **Superclass patterns** - Are all "vehicles" performing poorly?
3. **Small objects** - 32x32 images make some classes harder
4. **Similar names** - Model might confuse semantically similar words

## Next Steps After Analysis

Based on results, consider:
- **Data augmentation** for worst-performing classes
- **Longer training** if errors are random
- **Better encoder** if visual features are the issue
- **Beam search** if caption generation is the bottleneck
