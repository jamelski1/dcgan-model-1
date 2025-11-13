#!/usr/bin/env python3
"""
Comprehensive error analysis for CIFAR-100 image captioning.

Analyzes model performance including:
- Confusion matrix across all 100 classes
- Per-superclass performance breakdown
- Most confused class pairs
- Per-class accuracy analysis
- Sample visualizations of successes and failures
"""

import os
import sys
import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import make_splits, collate_pad, decode_ids
from models.encoders import EncoderCNN, DiscriminatorEncoder, ResNet18Encoder, HybridEncoder
from models.decoders import DecoderGRU
from models.gan import Discriminator
from models.attention import SpatialEncoder, AttentionDecoderGRU


# CIFAR-100 superclass mappings
CIFAR100_SUPERCLASSES = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


def get_superclass_mapping(class_names: List[str]) -> Dict[str, str]:
    """Create mapping from fine class to superclass."""
    class_to_super = {}
    for superclass, classes in CIFAR100_SUPERCLASSES.items():
        for cls in classes:
            class_to_super[cls] = superclass
    return class_to_super


def load_model(ckpt_path: str, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    stoi, itos = ckpt["vocab"]
    vocab_size = len(itos)
    args_dict = ckpt.get("args", {})

    feat_dim = args_dict.get("feat_dim", 256)
    ndf = args_dict.get("ndf", 64)
    hid = args_dict.get("hid", 256)
    emb = args_dict.get("emb", 128)
    encoder_type = args_dict.get("encoder_type", "cnn")
    use_attention = args_dict.get("use_attention", False)
    num_heads = args_dict.get("num_heads", 8)
    dropout = args_dict.get("dropout", 0.1)

    # Create encoder
    has_enc = "enc" in ckpt
    has_enc_disc = "enc_disc" in ckpt

    if has_enc_disc or encoder_type in ("dcgan", "dcgan_sn"):
        use_spectral_norm = (encoder_type == "dcgan_sn")
        discriminator = Discriminator(ndf=ndf, use_spectral_norm=use_spectral_norm)

        if use_attention:
            # Spatial encoder for attention mechanism
            encoder = SpatialEncoder(
                encoder=DiscriminatorEncoder(
                    discriminator=discriminator,
                    feat_dim=feat_dim,
                    ndf=ndf
                ),
                feat_dim=feat_dim,
                ndf=ndf
            ).to(device)
        else:
            # Standard global pooling encoder
            encoder = DiscriminatorEncoder(
                discriminator=discriminator,
                feat_dim=feat_dim,
                ndf=ndf
            ).to(device)

        enc_state = ckpt.get("enc_disc", ckpt.get("enc"))
        strict = not has_enc_disc
        encoder.load_state_dict(enc_state, strict=strict)
        encoder_desc = f"DCGAN {'spatial ' if use_attention else ''}encoder ({'SN' if use_spectral_norm else 'BN'})"
        print(f"Loaded {encoder_desc}")
    elif has_enc or encoder_type == "cnn":
        if use_attention:
            raise ValueError("Attention mechanism not supported with CNN encoder")
        encoder = EncoderCNN(feat_dim=feat_dim).to(device)
        encoder.load_state_dict(ckpt["enc"])
        print("Loaded baseline CNN encoder")
    elif encoder_type == "hybrid":
        # Hybrid encoder: Frozen DCGAN + ResNet18
        if not use_attention:
            raise ValueError("Hybrid encoder requires attention mechanism")

        # Create DCGAN spatial encoder (frozen)
        discriminator = Discriminator(ndf=ndf, use_spectral_norm=True)
        dcgan_spatial = SpatialEncoder(
            encoder=DiscriminatorEncoder(
                discriminator=discriminator,
                feat_dim=feat_dim,
                ndf=ndf,
                freeze=True
            ),
            feat_dim=feat_dim,
            ndf=ndf
        )

        # Create ResNet18 encoder
        resnet_encoder = ResNet18Encoder(out_channels=512, pretrained=False)

        # Combine into hybrid encoder
        encoder = HybridEncoder(
            dcgan_encoder=dcgan_spatial,
            resnet_encoder=resnet_encoder,
            dcgan_channels=feat_dim,
            resnet_channels=512
        ).to(device)

        # Load encoder state
        enc_state = ckpt.get("enc")
        if enc_state is None:
            raise KeyError("Checkpoint missing 'enc' key")
        encoder.load_state_dict(enc_state, strict=False)
        print(f"Loaded HybridEncoder (DCGAN {feat_dim}ch + ResNet18 512ch = 768ch)")
    else:
        raise ValueError("Cannot determine encoder type from checkpoint")

    # Determine spatial feature dimension for decoder
    spatial_feat_dim = 768 if encoder_type == "hybrid" else feat_dim

    # Create decoder based on attention flag
    if use_attention:
        decoder = AttentionDecoderGRU(
            spatial_feat_dim=spatial_feat_dim,
            vocab_size=vocab_size,
            hid=hid,
            emb=emb,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)
        print(f"Loaded AttentionDecoderGRU with {num_heads} heads")
    else:
        decoder = DecoderGRU(
            feat_dim=feat_dim,
            vocab_size=vocab_size,
            hid=hid,
            emb=emb
        ).to(device)
        print("Loaded standard DecoderGRU")
    decoder.load_state_dict(ckpt["dec"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, (stoi, itos)


def analyze_predictions(
    encoder,
    decoder,
    test_loader,
    stoi,
    itos,
    class_names,
    device,
    max_len=8
):
    """
    Analyze predictions and gather statistics.

    Returns:
        Dictionary with comprehensive analysis results
    """
    bos_id = stoi["<bos>"]
    eos_id = stoi["<eos>"]

    # Initialize tracking structures
    confusion_matrix = np.zeros((100, 100), dtype=int)
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confused_pairs = Counter()

    # Store sample predictions
    all_predictions = []
    failures = []
    successes = []

    print("Analyzing predictions...")
    with torch.no_grad():
        for images, targets, labels in tqdm(test_loader, desc="Processing"):
            images = images.to(device)

            # Generate predictions
            features = encoder(images)
            gen_ids = decoder.generate(features, bos_id, eos_id, max_len=max_len)

            for i in range(images.size(0)):
                true_label = labels[i].item()
                true_class = class_names[true_label]

                # Decode prediction
                pred_caption = decode_ids(gen_ids[i].tolist(), itos)
                pred_words = pred_caption.split()

                # Check if true class name appears in prediction
                correct = true_class in pred_words

                # Update per-class statistics
                per_class_correct[true_class] += correct
                per_class_total[true_class] += 1

                # Try to extract predicted class from caption
                pred_class = None
                for word in pred_words:
                    if word in class_names:
                        pred_class = word
                        break

                if pred_class:
                    pred_label = class_names.index(pred_class)
                    confusion_matrix[true_label, pred_label] += 1

                    if pred_class != true_class:
                        confused_pairs[(true_class, pred_class)] += 1
                else:
                    # No valid class predicted
                    confusion_matrix[true_label, true_label] += correct

                # Store example
                example = {
                    'image_idx': len(all_predictions),
                    'true_label': true_label,
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'pred_caption': pred_caption,
                    'correct': correct,
                    'image': images[i].cpu(),
                }
                all_predictions.append(example)

                if correct:
                    successes.append(example)
                else:
                    failures.append(example)

    return {
        'confusion_matrix': confusion_matrix,
        'per_class_accuracy': {cls: per_class_correct[cls] / per_class_total[cls]
                               for cls in class_names if per_class_total[cls] > 0},
        'per_class_total': dict(per_class_total),
        'confused_pairs': confused_pairs.most_common(20),
        'all_predictions': all_predictions,
        'successes': successes[:20],  # Top 20 successes
        'failures': failures[:20],  # Top 20 failures
    }


def analyze_superclass_performance(
    per_class_accuracy: Dict[str, float],
    class_names: List[str]
) -> Dict[str, Dict]:
    """Analyze performance by superclass."""
    class_to_super = get_superclass_mapping(class_names)

    superclass_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'classes': []})

    for cls, acc in per_class_accuracy.items():
        if cls in class_to_super:
            super_cls = class_to_super[cls]
            superclass_stats[super_cls]['classes'].append((cls, acc))
            superclass_stats[super_cls]['total'] += 1

    # Calculate superclass averages
    superclass_accuracy = {}
    for super_cls, stats in superclass_stats.items():
        avg_acc = np.mean([acc for _, acc in stats['classes']])
        superclass_accuracy[super_cls] = {
            'accuracy': avg_acc,
            'num_classes': stats['total'],
            'classes': sorted(stats['classes'], key=lambda x: x[1])  # Sort by accuracy
        }

    return superclass_accuracy


def visualize_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: str,
    top_n: int = 20
):
    """Visualize top N most confused classes."""
    # Get top N classes by total confusions
    row_sums = confusion_matrix.sum(axis=1)
    top_indices = np.argsort(row_sums)[-top_n:][::-1]

    # Extract submatrix
    sub_matrix = confusion_matrix[np.ix_(top_indices, top_indices)]
    sub_names = [class_names[i] for i in top_indices]

    # Create visualization
    plt.figure(figsize=(14, 12))
    sns.heatmap(sub_matrix, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=sub_names, yticklabels=sub_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Top {top_n} Most Confused Classes')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def visualize_sample_predictions(
    examples: List[Dict],
    output_path: str,
    title: str,
    n_samples: int = 10
):
    """Visualize sample predictions with images and captions."""
    n_samples = min(n_samples, len(examples))

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    for idx in range(n_samples):
        ex = examples[idx]
        ax = axes[idx]

        # Denormalize image
        img = ex['image'].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)

        ax.imshow(img)
        ax.axis('off')

        # Add caption
        pred_text = f"Pred: {ex['pred_caption'][:30]}"
        true_text = f"True: {ex['true_class']}"
        status = "✓" if ex['correct'] else "✗"

        ax.set_title(f"{status} {true_text}\n{pred_text}",
                    fontsize=9, pad=5)

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample predictions to {output_path}")


def create_performance_report(
    analysis_results: Dict,
    superclass_analysis: Dict,
    class_names: List[str],
    output_dir: str
):
    """Create comprehensive text report."""
    report_path = os.path.join(output_dir, "analysis_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CIFAR-100 IMAGE CAPTIONING - ERROR ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        per_class_acc = analysis_results['per_class_accuracy']
        overall_acc = np.mean(list(per_class_acc.values()))

        f.write(f"OVERALL PERFORMANCE\n")
        f.write(f"-" * 40 + "\n")
        f.write(f"Overall Label Accuracy: {overall_acc:.2%}\n")
        f.write(f"Classes Analyzed: {len(per_class_acc)}\n\n")

        # Best performing classes
        sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
        f.write(f"TOP 10 BEST PERFORMING CLASSES\n")
        f.write(f"-" * 40 + "\n")
        for i, (cls, acc) in enumerate(sorted_classes[:10], 1):
            f.write(f"{i:2d}. {cls:20s} {acc:.2%}\n")
        f.write("\n")

        # Worst performing classes
        f.write(f"TOP 10 WORST PERFORMING CLASSES\n")
        f.write(f"-" * 40 + "\n")
        for i, (cls, acc) in enumerate(sorted_classes[-10:], 1):
            f.write(f"{i:2d}. {cls:20s} {acc:.2%}\n")
        f.write("\n")

        # Most confused pairs
        f.write(f"TOP 10 MOST CONFUSED CLASS PAIRS\n")
        f.write(f"-" * 40 + "\n")
        for i, ((true_cls, pred_cls), count) in enumerate(analysis_results['confused_pairs'][:10], 1):
            f.write(f"{i:2d}. {true_cls:15s} -> {pred_cls:15s} ({count:3d} times)\n")
        f.write("\n")

        # Superclass performance
        f.write(f"PERFORMANCE BY SUPERCLASS\n")
        f.write(f"-" * 40 + "\n")
        sorted_super = sorted(superclass_analysis.items(),
                             key=lambda x: x[1]['accuracy'],
                             reverse=True)
        for super_cls, stats in sorted_super:
            f.write(f"{super_cls:35s} {stats['accuracy']:.2%} ({stats['num_classes']} classes)\n")
        f.write("\n")

    print(f"Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive error analysis")
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for evaluation")
    parser.add_argument("--max_len", type=int, default=8,
                       help="Maximum caption length")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Root directory for CIFAR-100 data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    encoder, decoder, vocab = load_model(args.ckpt, device)
    stoi, itos = vocab

    # Load test set
    print("Loading test dataset...")
    _, _, test_set, _ = make_splits(max_len=args.max_len, data_root=args.data_root)
    test_set.dataset.stoi, test_set.dataset.itos = stoi, itos

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_pad
    )

    class_names = test_set.dataset.fine_names

    # Run analysis
    analysis_results = analyze_predictions(
        encoder, decoder, test_loader, stoi, itos, class_names, device, args.max_len
    )

    # Analyze superclass performance
    superclass_analysis = analyze_superclass_performance(
        analysis_results['per_class_accuracy'],
        class_names
    )

    # Create visualizations
    print("\nGenerating visualizations...")

    # Confusion matrix
    visualize_confusion_matrix(
        analysis_results['confusion_matrix'],
        class_names,
        os.path.join(args.output_dir, "confusion_matrix.png"),
        top_n=20
    )

    # Sample successes
    if analysis_results['successes']:
        visualize_sample_predictions(
            analysis_results['successes'],
            os.path.join(args.output_dir, "sample_successes.png"),
            "Sample Successful Predictions",
            n_samples=10
        )

    # Sample failures
    if analysis_results['failures']:
        visualize_sample_predictions(
            analysis_results['failures'],
            os.path.join(args.output_dir, "sample_failures.png"),
            "Sample Failed Predictions",
            n_samples=10
        )

    # Create text report
    create_performance_report(
        analysis_results,
        superclass_analysis,
        class_names,
        args.output_dir
    )

    # Save detailed results to JSON
    json_results = {
        'overall_accuracy': float(np.mean(list(analysis_results['per_class_accuracy'].values()))),
        'per_class_accuracy': analysis_results['per_class_accuracy'],
        'confused_pairs': [(true_cls, pred_cls, int(count))
                          for (true_cls, pred_cls), count in analysis_results['confused_pairs']],
        'superclass_performance': {
            k: {'accuracy': float(v['accuracy']),
                'num_classes': v['num_classes'],
                'classes': [(cls, float(acc)) for cls, acc in v['classes']]}
            for k, v in superclass_analysis.items()
        }
    }

    json_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved detailed results to {json_path}")

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Overall Label Accuracy: {json_results['overall_accuracy']:.2%}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
