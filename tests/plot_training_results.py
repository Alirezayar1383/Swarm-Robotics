"""
Standalone plotting script for curriculum training results.

Reads the saved eval_log.npz files from each stage_N_logs/ directory
and generates all the figures needed for the thesis defense.

Usage:
    python plot_training_results.py
    python plot_training_results.py --videos_dir ../videos --smooth 5
"""

import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================================
# Argument parsing
# ============================================================================
parser = argparse.ArgumentParser(description='Plot curriculum training results.')
parser.add_argument('--videos_dir', type=str, default=None,
                    help='Directory containing stage_N_logs folders '
                         '(default: ../videos relative to this script)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Directory to save plots (default: same as videos_dir)')
parser.add_argument('--smooth', type=int, default=0,
                    help='Rolling-mean window size (0 = no smoothing)')
parser.add_argument('--dpi', type=int, default=150,
                    help='DPI for saved figures')
args = parser.parse_args()

# ============================================================================
# Resolve paths
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

if args.videos_dir is None:
    VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'videos')
else:
    VIDEOS_DIR = os.path.abspath(args.videos_dir)

if args.output_dir is None:
    OUTPUT_DIR = VIDEOS_DIR
else:
    OUTPUT_DIR = os.path.abspath(args.output_dir)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Reading from:  {VIDEOS_DIR}")
print(f"Saving plots:  {OUTPUT_DIR}")

# ============================================================================
# Load stage config
# ============================================================================
config_path = os.path.join(PROJECT_ROOT, 'configs', 'phase9_config.yaml')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    STAGES = cfg.get('stages', [])
else:
    print(f"⚠️  Config not found at {config_path}. Using default STAGES.")
    STAGES = [(4, 4, 500), (6, 6, 800), (8, 8, 1200),
              (10, 10, 1600), (12, 12, 2000), (15, 15, 2500), (18, 18, 3000)]

print(f"Loaded {len(STAGES)} stages from config.")

# ============================================================================
# Discover available stage logs
# ============================================================================
stage_dirs = []
for i in range(1, len(STAGES) + 1):
    d = os.path.join(VIDEOS_DIR, f"stage_{i}_logs")
    npz_path = os.path.join(d, 'eval_log.npz')
    if os.path.exists(npz_path):
        stage_dirs.append((i, d, npz_path))
    else:
        # stop scanning at first missing stage
        break

if not stage_dirs:
    print("❌ No stage logs found. Aborting.")
    sys.exit(1)

print(f"Found {len(stage_dirs)} stage logs:")
for i, d, p in stage_dirs:
    print(f"  Stage {i}: {p}")

# ============================================================================
# Load data from each stage
# ============================================================================
def load_stage_log(npz_path):
    """Load an npz file and convert to a dict of numpy arrays."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}

stages_data = []
for i, d, p in stage_dirs:
    log = load_stage_log(p)
    stages_data.append({
        'idx': i,
        'path': p,
        'episode': np.array(log.get('episode', [])),
        'coverage': np.array(log.get('coverage', [])),
        'avg_q': np.array(log.get('avg_q', [])),
        'loss': np.array(log.get('loss', [])),
        'lam': np.array(log.get('lam', [])),
        'overlap': np.array(log.get('overlap', [])),
        'fstay': np.array(log.get('fstay', [])),
        'td_error': np.array(log.get('td_error', [])),
        'episode_length': np.array(log.get('episode_length', [])),
        'frontier_count': np.array(log.get('frontier_count', [])),
        'comm_events': np.array(log.get('comm_events', [])),
    })

num_stages = len(stages_data)
print(f"\nLoaded {num_stages} stages.")
for s in stages_data:
    print(f"  Stage {s['idx']}: {len(s['episode'])} eval points, "
          f"final cov={s['coverage'][-1]*100:.1f}%, "
          f"final AvgQ={s['avg_q'][-1]:+.3f}")

# ============================================================================
# Helper: rolling mean
# ============================================================================
def rolling_mean(data, window):
    data = np.asarray(data, dtype=float)
    if window <= 1 or len(data) < window:
        return data
    kernel = np.ones(window) / window
    # use 'same' to preserve length; edge artifacts are acceptable
    return np.convolve(data, kernel, mode='same')

def maybe_smooth(data, window):
    if window > 1:
        return rolling_mean(data, window)
    return np.asarray(data, dtype=float)

# ============================================================================
# Plot 1: Main summary — AvgQ, Loss, TD Error per stage
# ============================================================================
print("\n📊 Plot 1: curriculum_summary.png")
fig, axes = plt.subplots(3, num_stages, figsize=(4 * num_stages, 12))
if num_stages == 1:
    axes = axes.reshape(3, 1)

for i, s in enumerate(stages_data):
    w, h = STAGES[s['idx'] - 1][0], STAGES[s['idx'] - 1][1]

    axes[0, i].plot(s['episode'], maybe_smooth(s['avg_q'], args.smooth),
                    'r-', linewidth=2)
    axes[0, i].set_title(f"Stage {s['idx']} ({w}×{h}) Avg Q", fontsize=11)
    axes[0, i].set_xlabel("Episode")
    axes[0, i].set_ylabel("Avg Q")
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].axhline(0, color='gray', linewidth=0.6, linestyle='--')

    axes[1, i].plot(s['episode'], maybe_smooth(s['loss'], args.smooth),
                    'b-', linewidth=2)
    axes[1, i].set_title(f"Stage {s['idx']} Training Loss", fontsize=11)
    axes[1, i].set_xlabel("Episode")
    axes[1, i].set_ylabel("Loss")
    axes[1, i].grid(True, alpha=0.3)

    axes[2, i].plot(s['episode'], maybe_smooth(s['td_error'], args.smooth),
                    'g-', linewidth=2)
    axes[2, i].set_title(f"Stage {s['idx']} TD Error (Exec)", fontsize=11)
    axes[2, i].set_xlabel("Episode")
    axes[2, i].set_ylabel("TD Error")
    axes[2, i].grid(True, alpha=0.3)

smooth_str = f" [smoothed, w={args.smooth}]" if args.smooth > 1 else ""
fig.suptitle(f"Curriculum Training Summary (Avg Q, Loss, TD Error){smooth_str}",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "curriculum_summary.png"), dpi=args.dpi)
plt.close(fig)

# ============================================================================
# Plot 2: Curriculum Transfer — AvgQ start vs end per stage (KEY PLOT)
# ============================================================================
print("📊 Plot 2: curriculum_transfer.png")
stage_labels = [f"Stage {s['idx']}\n({STAGES[s['idx']-1][0]}×{STAGES[s['idx']-1][1]})"
                for s in stages_data]
start_q = [s['avg_q'][0] if len(s['avg_q']) > 0 else 0 for s in stages_data]
end_q = [s['avg_q'][-1] if len(s['avg_q']) > 0 else 0 for s in stages_data]

fig, ax = plt.subplots(figsize=(max(10, 1.6 * num_stages), 6))
x = np.arange(num_stages)
width = 0.35
bars1 = ax.bar(x - width / 2, start_q, width,
               label='Start of Stage', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width / 2, end_q, width,
               label='End of Stage', color='crimson', edgecolor='black')

ax.set_xlabel('Training Stage', fontsize=12)
ax.set_ylabel('Avg Q (mean across agents)', fontsize=12)
ax.set_title('Curriculum Learning: AvgQ Transfer Between Stages',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(stage_labels, fontsize=10)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars1, start_q):
    h = bar.get_height()
    offset = -0.05 if h < 0 else 0.02
    ax.text(bar.get_x() + bar.get_width() / 2., h + offset,
            f'{h:+.3f}', ha='center',
            va='top' if h < 0 else 'bottom', fontsize=9)
for bar, val in zip(bars2, end_q):
    h = bar.get_height()
    offset = -0.05 if h < 0 else 0.02
    ax.text(bar.get_x() + bar.get_width() / 2., h + offset,
            f'{h:+.3f}', ha='center',
            va='top' if h < 0 else 'bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "curriculum_transfer.png"), dpi=args.dpi)
plt.close(fig)

# ============================================================================
# Plot 3: Coverage per stage
# ============================================================================
print("📊 Plot 3: coverage_per_stage.png")
fig, ax = plt.subplots(figsize=(12, 6))
for s in stages_data:
    w, h = STAGES[s['idx'] - 1][0], STAGES[s['idx'] - 1][1]
    ax.plot(s['episode'], s['coverage'] * 100,
            marker='o', markersize=3, linewidth=1.5,
            label=f"Stage {s['idx']} ({w}×{h})")
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Coverage (%)', fontsize=12)
ax.set_title('Coverage Progression Across Stages', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "coverage_per_stage.png"), dpi=args.dpi)
plt.close(fig)

# ============================================================================
# Plot 4: Combined metric comparison across stages (bar chart)
# ============================================================================
print("📊 Plot 4: stage_comparison.png")
final_losses = [s['loss'][-1] if len(s['loss']) > 0 else 0 for s in stages_data]
final_td = [s['td_error'][-1] if len(s['td_error']) > 0 else 0 for s in stages_data]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(range(num_stages), end_q, color='crimson', edgecolor='black')
axes[0].set_xticks(range(num_stages))
axes[0].set_xticklabels([f"S{s['idx']}" for s in stages_data])
axes[0].set_ylabel('Final AvgQ', fontsize=11)
axes[0].set_title('Final AvgQ per Stage', fontsize=12, fontweight='bold')
axes[0].axhline(0, color='gray', linewidth=0.8, linestyle='--')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(range(num_stages), final_losses, color='steelblue', edgecolor='black')
axes[1].set_xticks(range(num_stages))
axes[1].set_xticklabels([f"S{s['idx']}" for s in stages_data])
axes[1].set_ylabel('Final Loss', fontsize=11)
axes[1].set_title('Final Loss per Stage', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

axes[2].bar(range(num_stages), final_td, color='seagreen', edgecolor='black')
axes[2].set_xticks(range(num_stages))
axes[2].set_xticklabels([f"S{s['idx']}" for s in stages_data])
axes[2].set_ylabel('Final TD Error', fontsize=11)
axes[2].set_title('Final TD Error per Stage', fontsize=12, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

fig.suptitle('Stage-level Final Metrics Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stage_comparison.png"), dpi=args.dpi)
plt.close(fig)

# ============================================================================
# Plot 5: Smoothed dynamics (AvgQ + Loss)
# ============================================================================
print("📊 Plot 5: smoothed_dynamics.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

window = max(args.smooth, 3)
for s in stages_data:
    q_smooth = rolling_mean(s['avg_q'], window)
    axes[0].plot(range(len(q_smooth)), q_smooth,
                 linewidth=1.8, label=f"Stage {s['idx']}")
axes[0].set_xlabel('Evaluation Index', fontsize=11)
axes[0].set_ylabel('AvgQ (smoothed)', fontsize=11)
axes[0].set_title(f'Smoothed AvgQ Trajectory (window={window})',
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for s in stages_data:
    l_smooth = rolling_mean(s['loss'], window)
    axes[1].plot(range(len(l_smooth)), l_smooth,
                 linewidth=1.8, label=f"Stage {s['idx']}")
axes[1].set_xlabel('Evaluation Index', fontsize=11)
axes[1].set_ylabel('Loss (smoothed)', fontsize=11)
axes[1].set_title(f'Smoothed Loss Trajectory (window={window})',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle(f'Smoothed Training Dynamics (Window={window})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "smoothed_dynamics.png"), dpi=args.dpi)
plt.close(fig)

# ============================================================================
# Plot 6: Performance metrics — λ (Time Save Factor) per stage
# ============================================================================
print("📊 Plot 6: lambda_per_stage.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for s in stages_data:
    w, h = STAGES[s['idx'] - 1][0], STAGES[s['idx'] - 1][1]
    axes[0].plot(s['episode'], s['lam'],
                 marker='o', markersize=3, linewidth=1.5,
                 label=f"Stage {s['idx']} ({w}×{h})")
axes[0].set_xlabel('Episode', fontsize=11)
axes[0].set_ylabel('λ (Time Save Factor)', fontsize=11)
axes[0].set_title('λ Progression per Stage', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9, loc='upper right')
axes[0].grid(True, alpha=0.3)

for s in stages_data:
    w, h = STAGES[s['idx'] - 1][0], STAGES[s['idx'] - 1][1]
    axes[1].plot(s['episode'], s['overlap'],
                 marker='s', markersize=3, linewidth=1.5,
                 label=f"Stage {s['idx']} ({w}×{h})")
axes[1].set_xlabel('Episode', fontsize=11)
axes[1].set_ylabel('O (Overlap)', fontsize=11)
axes[1].set_title('Overlap Progression per Stage', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9, loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lambda_per_stage.png"), dpi=args.dpi)
plt.close(fig)

# ============================================================================
# Plot 7: FStay (Forced Stay) and TD Error
# ============================================================================
print("📊 Plot 7: fstay_and_td.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for s in stages_data:
    w, h = STAGES[s['idx'] - 1][0], STAGES[s['idx'] - 1][1]
    axes[0].plot(s['episode'], maybe_smooth(s['fstay'], window),
                 linewidth=1.8, label=f"Stage {s['idx']} ({w}×{h})")
axes[0].set_xlabel('Episode', fontsize=11)
axes[0].set_ylabel('FStay (smoothed)', fontsize=11)
axes[0].set_title('Forced Stay Events per Stage', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for s in stages_data:
    w, h = STAGES[s['idx'] - 1][0], STAGES[s['idx'] - 1][1]
    axes[1].plot(s['episode'], maybe_smooth(s['td_error'], window),
                 linewidth=1.8, label=f"Stage {s['idx']} ({w}×{h})")
axes[1].set_xlabel('Episode', fontsize=11)
axes[1].set_ylabel('TD Error (smoothed)', fontsize=11)
axes[1].set_title('Execution TD Error per Stage', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Execution Quality Metrics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fstay_and_td.png"), dpi=args.dpi)
plt.close(fig)

# ============================================================================
# Plot 8: Curriculum Transfer Summary — Combined view
# ============================================================================
print("📊 Plot 8: curriculum_full_summary.png")

# Build a continuous AvgQ trajectory across stages
continuous_avg_q = []
continuous_episode = []
offset = 0
stage_boundaries = []
for s in stages_data:
    n = len(s['avg_q'])
    continuous_avg_q.extend(s['avg_q'])
    continuous_episode.extend(np.arange(offset, offset + n))
    offset += n
    stage_boundaries.append(offset)

continuous_avg_q = np.array(continuous_avg_q)
continuous_episode = np.array(continuous_episode)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top: continuous AvgQ
axes[0].plot(continuous_episode, continuous_avg_q, 'b-', linewidth=1.5, alpha=0.7)
axes[0].plot(continuous_episode, rolling_mean(continuous_avg_q, 5),
             'r-', linewidth=2.5, label='Smoothed (w=5)')
axes[0].axhline(0, color='gray', linewidth=0.6, linestyle='--')
for b in stage_boundaries[:-1]:
    axes[0].axvline(b, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
axes[0].set_xlabel('Cumulative Evaluation Index', fontsize=11)
axes[0].set_ylabel('Avg Q', fontsize=11)
axes[0].set_title('Continuous AvgQ Across All Curriculum Stages',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Bottom: stage-wise AvgQ trajectories (aligned to their own x)
for s in stages_data:
    w, h = STAGES[s['idx'] - 1][0], STAGES[s['idx'] - 1][1]
    axes[1].plot(range(len(s['avg_q'])), s['avg_q'],
                 linewidth=1.8, alpha=0.85,
                 label=f"Stage {s['idx']} ({w}×{h})")
axes[1].axhline(0, color='gray', linewidth=0.6, linestyle='--')
axes[1].set_xlabel('Evaluation Index (within stage)', fontsize=11)
axes[1].set_ylabel('Avg Q', fontsize=11)
axes[1].set_title('AvgQ Trajectory per Stage (Aligned)',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "curriculum_full_summary.png"), dpi=args.dpi)
plt.close(fig)

# ============================================================================
# Print final summary table
# ============================================================================
print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)
print(f"{'Stage':>6} | {'Map':>8} | {'#Evals':>7} | {'Start Q':>10} | {'End Q':>10} | "
      f"{'Final Loss':>10} | {'Final TD':>10} | {'Final Cov':>10}")
print("-" * 100)
for s in stages_data:
    w, h = STAGES[s['idx'] - 1][0], STAGES[s['idx'] - 1][1]
    n = len(s['avg_q'])
    start_q = s['avg_q'][0] if n > 0 else 0
    end_q_val = s['avg_q'][-1] if n > 0 else 0
    final_loss = s['loss'][-1] if len(s['loss']) > 0 else 0
    final_td = s['td_error'][-1] if len(s['td_error']) > 0 else 0
    final_cov = s['coverage'][-1] if len(s['coverage']) > 0 else 0
    print(f"{s['idx']:>6} | {w:>3}×{h:<3} | {n:>7} | {start_q:>+10.4f} | "
          f"{end_q_val:>+10.4f} | {final_loss:>10.4f} | {final_td:>10.4f} | "
          f"{final_cov*100:>9.1f}%")
print("=" * 100)

print(f"\n✅ All plots saved to: {OUTPUT_DIR}")
print("Generated files:")
for fname in ["curriculum_summary.png", "curriculum_transfer.png",
              "coverage_per_stage.png", "stage_comparison.png",
              "smoothed_dynamics.png", "lambda_per_stage.png",
              "fstay_and_td.png", "curriculum_full_summary.png"]:
    full = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(full):
        print(f"  ✅ {fname}")