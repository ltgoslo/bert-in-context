import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cmap import Colormap
from matplotlib.colors import ListedColormap

cm = Colormap(["#FFCEC7", "#E5E5E5", "#CADAFC"]).to_mpl()

# Define the sequence lengths and filename pattern
seq_lengths = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288]
filename_patterns = ["result_haystack_{}_opt-1.3b", "result_haystack_{}_deberta-xxlarge-fixed"]

def get_matrix(filename_pattern):

    # Initialize dictionary to store the scores
    results = {seq_len: {} for seq_len in seq_lengths}

    # Loop through each file corresponding to a sequence length
    for seq_len in seq_lengths:
        filename = filename_pattern.format(seq_len)
        if not os.path.exists(filename):
            for fraction in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                results[seq_len][fraction] = [0.0]
            continue

        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                # Split the line and extract the necessary fields
                parts = line.strip().split(',')
                fraction = float(parts[2])
                score = int(parts[3])

                # Initialize or update the dictionary for this fraction
                if fraction not in results[seq_len]:
                    results[seq_len][fraction] = []
                results[seq_len][fraction].append(score)

    # Calculate the average score for each seq_len and fraction
    avg_scores = {seq_len: {frac: np.mean(scores) for frac, scores in fracs.items()}
                for seq_len, fracs in results.items()}

    # Prepare data for heatmap
    fractions = sorted(list({frac for fracs in results.values() for frac in fracs.keys()}))
    heatmap_data = np.array([[avg_scores[seq_len][frac] for frac in fractions] for seq_len in seq_lengths])

    return heatmap_data.T * 100, fractions

matrix_a, fractions = get_matrix(filename_patterns[0])
matrix_b, _ = get_matrix(filename_patterns[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'wspace': 0.1})

# Plot the first matrix
im1 = ax1.imshow(matrix_a, cmap=cm, aspect='auto', vmin=0, vmax=100)

# Plot the second matrix
im2 = ax2.imshow(matrix_b, cmap=cm, aspect='auto', vmin=0, vmax=100)

# Remove frames around the plots
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Write values on the heatmap
for i in range(len(seq_lengths)):
    for j in range(len(fractions)):
        # ax.text(i, j, f'{int(heatmap_data[i, j]*100 + 0.5)}', ha='center', va='center', color='black' if heatmap_data[i, j] > 0.5 else "white", fontsize=7)
        ax1.text(i, j, f'{int(matrix_a[j, i])}', ha='center', va='center', color='black', fontsize=6)
        ax2.text(i, j, f'{int(matrix_b[j, i])}', ha='center', va='center', color='black', fontsize=6)

# Set titles
ax1.set_title('OPT', pad=10, fontsize=13, fontweight='bold', fontname='Times New Roman')
ax2.set_title('DeBERTa', pad=10, fontsize=13, fontweight='bold', fontname='Times New Roman')

# Set x and y labels
ax1.set_ylabel('Needle position', labelpad=10, fontsize=9, fontweight='bold')
ax1.set_xlabel('Sequence length', labelpad=7.5, fontsize=9, fontweight='bold')
ax1.set_xticks([0, 2, 4, 6, 8, 10])  # Set major ticks
ax1.set_xticklabels([256, 512, 1024, 2048, 4096, 8192])
ax1.tick_params(axis='x', which='major', length=0, labelsize=8)  # Set font size for major ticks
ax1.set_xticks([1, 3, 5, 7, 9, 11], minor=True)  # Set minor ticks
ax1.set_xticklabels([384, 768, 1536, 3072, 6144, 12288], minor=True)
ax1.tick_params(axis='x', which='minor', labelsize=8, length=0)  # Set font size for minor ticks
ax1.set_yticks([0, 2, 4, 6, 8, 10])  # Set major ticks
ax1.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.tick_params(axis='y', which='major', length=0, labelsize=8)  # Set font size for major ticks
ax1.set_yticks([1, 3, 5, 7, 9], minor=True)  # Set minor ticks
ax1.set_yticklabels([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
ax1.tick_params(axis='y', which='minor', labelsize=8, length=0)  # Set font size for minor ticks
ax1.tick_params(axis='both', length=0)

ax2.set_xlabel('Sequence length', labelpad=7.5, fontsize=9, fontweight='bold')
ax2.set_xticks([0, 2, 4, 6, 8, 10])  # Set major ticks
ax2.set_xticklabels([256, 512, 1024, 2048, 4096, 8192])
ax2.tick_params(axis='x', which='major', length=0, labelsize=8)  # Set font size for major ticks
ax2.set_xticks([1, 3, 5, 7, 9, 11], minor=True)  # Set minor ticks
ax2.set_xticklabels([384, 768, 1536, 3072, 6144, 12288], minor=True)
ax2.tick_params(axis='x', which='minor', labelsize=8, length=0, labelcolor='black')  # Set font size for minor ticks
ax2.tick_params(axis='both', length=0)
ax2.set_yticks([0, 2, 4, 6, 8, 10])  # Set major ticks
ax2.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.tick_params(axis='y', which='major', length=0, labelsize=8)  # Set font size for major ticks
ax2.set_yticks([1, 3, 5, 7, 9], minor=True)  # Set minor ticks
ax2.set_yticklabels([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
ax2.tick_params(axis='y', which='minor', labelsize=8, length=0, labelcolor='black')  # Set font size for minor ticks
ax2.tick_params(axis='both', length=0)

for i in range(13):
    ax1.plot([i - 0.5,i - 0.5], [-1.0, 10.5], color='white', linewidth=3)

# Draw thick vertical red line
ax1.plot([6.5, 6.5], [-1.0, 10.5], color="#C9001E", linewidth=3)

# Add text annotations
ax1.text(6.25, -0.7, '⟵ Pretraining', fontsize=9, ha='right', color="red", fontweight='bold')
ax1.text(6.75, -0.7, 'Length generalization ⟶', fontsize=9, ha='left', color="red", fontweight='bold')

for i in range(13):
    ax2.plot([i - 0.5,i - 0.5], [-1.0, 10.5], color='white', linewidth=3)

# Draw thick vertical red line
ax2.plot([2.5, 2.5], [-1.0, 10.5], color="#C9001E", linewidth=3)

# Add text annotations
ax2.text(2.25, -0.7, '← Pretraining', fontsize=9, ha='right', color='red', fontweight='bold')
ax2.text(2.75, -0.7, 'Length generalization →', fontsize=9, ha='left', color='red', fontweight='bold')

bbox = ax2.get_position()
cbar_ax = fig.add_axes([bbox.x1 + 0.02, bbox.y0, 0.02, bbox.height * 0.91])
cbar = fig.colorbar(im2, cax=cbar_ax, )
cbar.outline.set_visible(False)
cbar.set_label('Accuracy', fontsize=9, fontweight='bold')
cbar.ax.tick_params(labelsize=8, length=0)

plt.subplots_adjust(left=0.05)
plt.savefig(f'haystack_heatmap.pdf')

print("Done")
