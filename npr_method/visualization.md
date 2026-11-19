
# NPR Visualization and Analysis

## 1. Objective

The objective of this notebook is to visualize and compare Neighboring Pixel Relationships (NPR) in real and AI-generated images.

The analysis focuses on the distributions of pixel differences across three spatial directions and three RGB channels.

It aims to determine whether real and fake images exhibit distinct NPR patterns, and whether these differences exceed the natural variability within each class.

## 2. Real vs Fake Histogram Comparison

For each RGB channel (R, G, B) and each spatial direction (horizontal, vertical, diagonal), the distributions of pixel differences are visualized using histograms.

This produces nine histogram pairs:

- Horizontal: R, G, B
- Vertical: R, G, B
- Diagonal: R, G, B

Each pair displays:
- The real image distribution.
- The fake image distribution.

All histograms use 16 bins over the fixed range [-255, 255], with density normalization.

Real and fake distributions are displayed vertically with shared axes, allowing direct visual comparison.

This first visualization provides a qualitative overview of the differences between the two classes.

## 3. Intra-class and Inter-class Variability

### 3.1. Motivation

Observed differences between real and fake distributions may result from the distinction between the classes, but they may also reflect differences in dataset composition.

To investigate this, we compare variability within each class against variability between classes.

### 3.2. Random Splitting

For each experiment, both datasets are independently shuffled and divided into two equally sized groups:

- Real 1 and Real 2.
- Fake 1 and Fake 2.

The experiment is repeated using 20 different random seeds.

For each seed, four comparisons are performed:

| Comparison | Purpose |
|---|---|
| Real 1 - Real 2 | Real intra-class variability |
| Fake 1 - Fake 2 | Fake intra-class variability |
| Real 1 - Fake 2 | Inter-class variability |
| Real 2 - Fake 1 | Inter-class variability |

### 3.3. Histogram Distance

For each group, a normalized histogram is computed using 16 fixed bins over [-255, 255].

The distance between two histograms is measured using the L1 distance:

$$
D(H_A,H_B)=\sum_{i=1}^{16}|H_A(i)-H_B(i)|
$$

A distance of 0 indicates identical distributions, while larger values indicate greater differences.

For each seed, this distance is computed independently for all four comparisons, across the nine channel-direction combinations.

### 3.4. Statistical Results

For each comparison and each channel-direction combination, the distances obtained across the 20 seeds are summarized using:

- **Mean distance:** average difference between the two distributions.
- **Standard deviation:** variability of the measured distance across random splits.

The mean measures the magnitude of the distributional differences, while the standard deviation indicates their sensitivity to dataset partitioning.

## 4. Heatmap Visualization

The final results are summarized in a heatmap.

- **Rows:** the nine channel-direction combinations.
- **Columns:** the four dataset comparisons.
- **Cell values:** mean histogram distances across the 20 seeds.
- **Color intensity:** magnitude of the mean distance.

This representation makes it possible to identify which channels and directions exhibit the strongest differences and to compare intra-class and inter-class variability at a glance.

## 5. Interpretation

The main hypothesis is that inter-class distances should be consistently greater than intra-class distances:

$$
D_{RF} > D_{RR}
\quad \text{and} \quad
D_{RF} > D_{FF}
$$

A consistent separation would suggest that NPR distributions contain information related to the distinction between real and AI-generated images.

However, this experiment does not establish that authenticity alone explains the observed differences. Other factors, such as image content, dataset composition, compression, and generation sources, may also contribute.

Further experiments with larger and more diverse datasets are needed to evaluate the robustness and generalization of these observations.