# Computational Robotics Computer Vision Project: NeRF-Nothing

## Team Members

- Luke Witten
- Jessica Brown

## Explanation of Neural Radiance Fields (NeRF)

A Neural Radiance Field (NeRF) is a neural network-based method of creating new, photorealistic depictions of 3D scenes. To train a NeRF, one inputs a collection of images, along with their pose and orientation, into a neural network. This network interprets them in 3D space by casting rays through the images, training a continuous 3D function that contains color values matching those projected from the training images.

To create a new image from the NeRF, a clever sampling method is used. Imagine drawing a conic region through the image that grows larger as it approaches the camera. At each step along the vector, colors are sampled within a frustrum and accumulated along with depth and opacity values, which are weighted by the distance from neighboring samples. This blending of color, density, and depth values determines the final value at each pixel.

![NeRF Visualization](LINK_TO_YOUR_IMAGE)

## Code Implementation

We used NerfStudio to create NeRFs using image and orientation data collected from the PolyCam app.

- NerfStudio Citation
- NerfStudio Citation
- NerfStudio Citation

![NeRF Sample Image](LINK_TO_YOUR_IMAGE)

Further modifications were made to how the renderers determine density, color, and depth values. The AccumulationRenderer now includes a light intensity correction according to the inverse square law, for more realistic color blending.

![Inverse Square Law Light Intensity](LINK_TO_YOUR_IMAGE)

Depth renderer changes include a weighted median for depth values and robust filtering to handle outliers in the dataset. Outliers are images whose RGB values do not match well with their pose, but dataset pruning helps mitigate this issue.

We also added gamma correction to our RGB renderer. This adjustment isn't theoretically necessary but does improve details in shadows and display quality.

![Gamma Correction](LINK_TO_YOUR_IMAGE)

### Jessica's Normalization Explanation

TODO: Jessica's explanation about normalization.

## Trained and Altered NeRF Representations

![NeRF Representation 1](LINK_TO_YOUR_IMAGE)

![NeRF Representation 2](LINK_TO_YOUR_IMAGE)

![NeRF Representation 3](LINK_TO_YOUR_IMAGE)
