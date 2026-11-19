# NPR Feature Extraction

## 1. Objective

The goal of the NPR (Neighboring Pixel Relationships) method is to capture local spatial relationships between neighboring pixels that may differ between real photographs and AI-generated images.

The method extracts statistical features from local pixel differences. It does not directly classify images.

## 2. Local Pixel Differences

All images are processed in **RGB color space**.

Each color channel (Red, Green, Blue) is processed independently.

Divide the image into non-overlapping $2 \times 2$ blocks:

$$
\begin{bmatrix}
A_1 & A_2 \\
A_3 & A_4
\end{bmatrix}
$$

For each block, use the top-left pixel $A_1$ as the reference and calculate three differences:

$$
d_1 = A_2 - A_1 \qquad \text{(horizontal)}
$$

$$
d_2 = A_3 - A_1 \qquad \text{(vertical)}
$$

$$
d_3 = A_4 - A_1 \qquad \text{(diagonal)}
$$

Repeat this operation for every complete $2 \times 2$ block in the image.

Keep the three collections of differences separate, since they represent different spatial relationships.

For images with odd dimensions, discard the final incomplete row or column.

## 3. Feature Extraction

For each difference collection, calculate four statistical features over **all blocks in the image**:

1. **Mean**
2. **Variance**
3. **Skewness**
4. **Kurtosis** (Fisher excess kurtosis)

Additionally, compute a **16-bin normalized histogram** of the differences.

The four statistics describe the overall distribution, while the histogram captures its shape in greater detail.

Histogram boundaries must remain consistent across all images, channels, and directions. They should be defined according to the numerical representation of the input images, not adapted independently to each image.

## 4. RGB Processing and Feature Vector

Apply the procedure independently to all three RGB channels.

For each channel, extract features for the three directions:

- Horizontal
- Vertical
- Diagonal

Each direction produces:

- 4 statistical features
- 16 histogram values

The total number of features is therefore:

$$
(4+16)\times 3\ \text{directions}\times 3\ \text{channels} = \boxed{180}
$$

Each image is represented by a fixed-length feature vector:

$$
\mathbf{x}_{NPR} \in \mathbb{R}^{180}
$$

The feature ordering must remain identical across all images.
