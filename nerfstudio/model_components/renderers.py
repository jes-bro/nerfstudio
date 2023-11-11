# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of renderers

Example:

.. code-block:: python

    field_outputs = field(ray_sampler)
    weights = ray_sampler.get_weights(field_outputs[FieldHeadNames.DENSITY])

    rgb_renderer = RGBRenderer()
    rgb = rgb_renderer(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

"""
import contextlib
import math
from typing import Generator, Literal, Optional, Tuple, Union

import nerfacc
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.utils import colors
from nerfstudio.utils.math import (components_from_spherical_harmonics,
                                   safe_normalize)

BackgroundColor = Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None


@contextlib.contextmanager
def background_color_override_context(mode: Float[Tensor, "3"]) -> Generator[None, None, None]:
    """Context manager for setting background mode."""
    global BACKGROUND_COLOR_OVERRIDE
    old_background_color = BACKGROUND_COLOR_OVERRIDE
    try:
        BACKGROUND_COLOR_OVERRIDE = mode
        yield
    finally:
        BACKGROUND_COLOR_OVERRIDE = old_background_color


class RGBRenderer(nn.Module):
    """Standard volumetric rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    def __init__(self, background_color: BackgroundColor = "random") -> None:
        super().__init__()
        self.background_color: BackgroundColor = background_color

    @classmethod
    def combine_rgb(
        cls,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BackgroundColor = "random",
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image.
        If background color is random, no BG color is added - as if the background was black!

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_rgb = nerfacc.accumulate_along_rays(
                weights[..., 0], values=rgb, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_rgb = torch.sum(weights * rgb, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if background_color == "random":
            # If background color is random, the predicted color is returned without blending,
            # as if the background color was black.
            return comp_rgb
        elif background_color == "last_sample":
            # Note, this is only supported for non-packed samples.
            background_color = rgb[..., -1, :]
        background_color = cls.get_background_color(background_color, shape=comp_rgb.shape, device=comp_rgb.device)

        assert isinstance(background_color, torch.Tensor)
        comp_rgb = comp_rgb + background_color * (1.0 - accumulated_weight)
        return comp_rgb

    @classmethod
    def get_background_color(
        cls, background_color: BackgroundColor, shape: Tuple[int, ...], device: torch.device
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        """Returns the RGB background color for a specified background color.
        Note:
            This function CANNOT be called for background_color being either "last_sample" or "random".

        Args:
            background_color: The background color specification. If a string is provided, it must be a valid color name.
            shape: Shape of the output tensor.
            device: Device on which to create the tensor.

        Returns:
            Background color as RGB.
        """
        assert background_color not in {"last_sample", "random"}
        assert shape[-1] == 3, "Background color must be RGB."
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color]
        assert isinstance(background_color, Tensor)

        # Ensure correct shape
        return background_color.expand(shape).to(device)

    def blend_background(
        self,
        image: Tensor,
        background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Blends the background color into the image if image is RGBA.
        Otherwise no blending is performed (we assume opacity of 1).

        Args:
            image: RGB/RGBA per pixel.
            opacity: Alpha opacity per pixel.
            background_color: Background color.

        Returns:
            Blended RGB.
        """
        if image.size(-1) < 4:
            return image

        rgb, opacity = image[..., :3], image[..., 3:]
        if background_color is None:
            background_color = self.background_color
            if background_color in {"last_sample", "random"}:
                background_color = "black"
        background_color = self.get_background_color(background_color, shape=rgb.shape, device=rgb.device)
        assert isinstance(background_color, torch.Tensor)
        return rgb * opacity + background_color.to(rgb.device) * (1 - opacity)

    def blend_background_for_loss_computation(
        self,
        pred_image: Tensor,
        pred_accumulation: Tensor,
        gt_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Blends a background color into the ground truth and predicted image for
        loss computation.

        Args:
            gt_image: The ground truth image.
            pred_image: The predicted RGB values (without background blending).
            pred_accumulation: The predicted opacity/ accumulation.
        Returns:
            A tuple of the predicted and ground truth RGB values.
        """
        background_color = self.background_color
        if background_color == "last_sample":
            background_color = "black"  # No background blending for GT
        elif background_color == "random":
            #background_color = torch.rand_like(pred_image)
            background_color = torch.rand(*pred_image.shape).to(dtype=pred_image.dtype, device=pred_image.device)
            pred_image = pred_image + background_color * (1.0 - pred_accumulation)
            # An example of a Screen blending mode
            #pred_image = 1 - (1 - pred_image) * (1 - background_color)
        gt_image = self.blend_background(gt_image, background_color=background_color)
        return pred_image, gt_image

    @staticmethod
    def apply_gamma_correction(rgb, gamma=2.2):
        """Applies gamma correction to the rgb values."""
        gamma_inv = 1.0 / gamma
        # Ensure that the RGB values are in the range [0, 1]
        rgb = torch.clamp(rgb, min=0.0, max=1.0)
        # Apply gamma correction
        return rgb.pow(gamma_inv)

    def forward(
        self,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
        background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.
            background_color: The background color to use for rendering.

        Returns:
            Outputs of rgb values.
        """

        """        
        if background_color is None:
            background_color = self.background_color

        if not self.training:
            rgb = torch.nan_to_num(rgb)
        rgb = self.combine_rgb(
            rgb, weights, background_color=background_color, ray_indices=ray_indices, num_rays=num_rays
        )
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        return rgb
        """
        if background_color is None:
            background_color = self.background_color

        if not self.training:
            rgb = torch.nan_to_num(rgb)
        
        rgb = self.combine_rgb(
            rgb, weights, background_color=background_color, ray_indices=ray_indices, num_rays=num_rays
        )
        
        # Apply gamma correction here
        rgb = self.apply_gamma_correction(rgb, gamma=2.2)
        
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        
        return rgb


class SHRenderer(nn.Module):
    """Render RGB value from spherical harmonics.

    Args:
        background_color: Background color as RGB. Uses random colors if None
        activation: Output activation.
    """

    def __init__(
        self,
        background_color: BackgroundColor = "random",
        activation: Optional[nn.Module] = nn.Sigmoid(),
    ) -> None:
        super().__init__()
        self.background_color: BackgroundColor = background_color
        self.activation = activation

    def forward(
        self,
        sh: Float[Tensor, "*batch num_samples coeffs"],
        directions: Float[Tensor, "*batch num_samples 3"],
        weights: Float[Tensor, "*batch num_samples 1"],
    ) -> Float[Tensor, "*batch 3"]:
        """Composite samples along ray and render color image

        Args:
            sh: Spherical harmonics coefficients for each sample
            directions: Sample direction
            weights: Weights for each sample

        Returns:
            Outputs of rgb values.
        """

        sh = sh.view(*sh.shape[:-1], 3, sh.shape[-1] // 3)

        levels = int(math.sqrt(sh.shape[-1]))
        components = components_from_spherical_harmonics(levels=levels, directions=directions)

        rgb = sh * components[..., None, :]  # [..., num_samples, 3, sh_components]
        rgb = torch.sum(rgb, dim=-1)  # [..., num_samples, 3]

        if self.activation is not None:
            rgb = self.activation(rgb)

        if not self.training:
            rgb = torch.nan_to_num(rgb)
        rgb = RGBRenderer.combine_rgb(rgb, weights, background_color=self.background_color)
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)

        return rgb



class AccumulationRenderer(nn.Module):
    """Accumulated value along a ray."""

    @classmethod
    def forward(
        cls,
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
        distances: Optional[Float[Tensor]] = None,
    ) -> Float[Tensor, "*bs 1"]:
        """Composite samples along ray and calculate accumulation.

        Args:
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.
            distances: Optional distance weights for attenuation.

        Returns:
            Outputs of accumulated values.
        """

        # Check if distances are not provided
        if distances is None:
            # Check if ray_indices and num_rays are provided for packed samples
            if ray_indices is not None and num_rays is not None:
                # Necessary for packed samples from volumetric ray sampler
                accumulation = nerfacc.accumulate_along_rays(
                    weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
                )
            else:
                # If not packed, sum the weights across the samples dimension
                accumulation = torch.sum(weights, dim=-2)
        
        else:
            # Distance attenuation should be applied here if distances are provided

            # Apply distance attenuation to weights
            # Assuming that distances are already squared; square them if they are not.
            # Adding a small epsilon to prevent division by zero errors.
            epsilon = 1e-7
            epsilon_tensor = torch.tensor(epsilon, device=distances.device, dtype=distances.dtype)

            # Compute attenuated weights by dividing by distances (with epsilon added for numerical stability)
            attenuated_weights = weights / (distances + epsilon_tensor)

            # If samples are packed, accumulate them according to their ray indices
            if ray_indices is not None and num_rays is not None:
                accumulation = nerfacc.accumulate_along_rays(
                    attenuated_weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
                )
            else:
                # If samples are not packed, directly sum the attenuated weights
                accumulation = torch.sum(attenuated_weights, dim=-2)

        # Return the accumulated value along the ray
        return accumulation



class DepthRenderer(nn.Module):
    """Renderer for calculating depth from ray data using various methods."""

    def __init__(self, method: Literal["median", "expected", "robust", "robust_weighted_median"] = "median") -> None:
        super().__init__()
        self.method = method  # Depth calculation method to be used.

    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*batch 1"]:
        """Composite weighted samples to calculate depth."""

        if self.method == "median":
            # Calculate median depth values from samples.
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2  # Midpoints of sample intervals.
            cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # Cumulative weights across samples.
            split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # 50% weight threshold.
            median_index = torch.searchsorted(cumulative_weights, split, side="left")  # Index where weight is 50%.
            median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # Clamp index within valid range.
            median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # Get median depth values.
            return median_depth

        if self.method == "expected":
            # Calculate expected depth values from samples.
            eps = 1e-10  # Small epsilon to avoid division by zero.
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2  # Midpoints of sample intervals.
            depth = (torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)).clamp(min=steps.min(), max=steps.max())
            # Clamped weighted sum of steps to get expected depth.
            return depth

        if self.method == "robust":
            # Calculate robust depth by trimming outliers.
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2  # Midpoints of sample intervals.
            trim_percent = 0.1  # Percentage of outliers to trim from each tail.
            lower_trim_index = int(weights.shape[-2] * trim_percent)  # Lower trim index.
            upper_trim_index = int(weights.shape[-2] * (1 - trim_percent))  # Upper trim index.
            sorted_weights, sorted_indices = torch.sort(weights, dim=-2)  # Sort weights for trimming.
            sorted_steps = torch.gather(steps, dim=-2, index=sorted_indices)  # Sort steps accordingly.
            trimmed_weights = sorted_weights[..., lower_trim_index:upper_trim_index, :]  # Trim weights.
            trimmed_steps = sorted_steps[..., lower_trim_index:upper_trim_index, :]  # Trim steps.
            trimmed_weight_sum = trimmed_weights.sum(dim=-2, keepdim=True)  # Sum of trimmed weights.
            depth = (trimmed_weights * trimmed_steps).sum(dim=-2) / trimmed_weight_sum  # Calculate trimmed mean depth.
            return depth.clamp(min=steps.min(), max=steps.max())  # Clamp to valid range.

        if self.method == "robust_weighted_median":
            # Calculate depth using a robust weighted median approach.
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2  # Midpoints of sample intervals.
            flat_weights = weights.view(-1)  # Flatten weights for processing.
            flat_steps = steps.view(-1)  # Flatten steps for processing.
            median = torch.median(flat_steps)  # Calculate median of steps.
            mad = torch.median(torch.abs(flat_steps - median))  # Compute median absolute deviation (MAD).
            robust_weights = torch.exp(-(torch.abs(flat_steps - median) / (mad + 1e-6))**2)  # Calculate robust weights.
            flat_weights *= robust_weights  # Apply robust weights to original weights.
            sorted_steps, sorted_indices = torch.sort(flat_steps)  # Sort steps for median calculation.
            sorted_weights = flat_weights[sorted_indices]  # Sort weights accordingly.
            cumulative_weights = torch.cumsum(sorted_weights, dim=0)  # Cumulative sorted weights.
            total_weight = cumulative_weights[-1]  # Total weight for median calculation.
            median_idx = torch.searchsorted(cumulative_weights, total_weight / 2)  # Index of weighted median.
           


class UncertaintyRenderer(nn.Module):
    """Calculate uncertainty along the ray."""

    @classmethod
    def forward(
        cls, betas: Float[Tensor, "*bs num_samples 1"], weights: Float[Tensor, "*bs num_samples 1"]
    ) -> Float[Tensor, "*bs 1"]:
        """Calculate uncertainty along the ray.

        Args:
            betas: Uncertainty betas for each sample.
            weights: Weights of each sample.

        Returns:
            Rendering of uncertainty.
        """
        uncertainty = torch.sum(weights * betas, dim=-2)
        return uncertainty


class SemanticRenderer(nn.Module):
    """Calculate semantics along the ray."""

    @classmethod
    def forward(
        cls,
        semantics: Float[Tensor, "*bs num_samples num_classes"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs num_classes"]:
        """Calculate semantics along the ray."""
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            return nerfacc.accumulate_along_rays(
                weights[..., 0], values=semantics, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            return torch.sum(weights * semantics, dim=-2)


class NormalsRenderer(nn.Module):
    """Calculate normals along the ray."""
    
    def forward(self, normals: torch.Tensor, weights: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Calculate normals along the ray.
        Args:
            normals: Normals for each sample [batch_size, num_samples, 3].
            weights: Weights of each sample [batch_size, num_samples, 1].
            normalize: Normalize normals.
        """
        # Get the batch size and the number of samples from the normals tensor.
        batch_size, num_samples, _ = normals.size()
        # Initialize a new tensor for the new weights, same shape as original weights.
        new_weights = torch.zeros_like(weights)

        # Compute new weights based on angles between all pairs of normals.
        for i in range(num_samples):  # Loop over each sample.
            for j in range(i + 1, num_samples):  # Compare with every other sample.
                # Calculate the weight based on the angle between pair of normals.
                angle_weight = angle_based_weighting(normals[:, i, :], normals[:, j, :])
                # Add this weight to both normals being compared.
                new_weights[:, i, :] += angle_weight
                new_weights[:, j, :] += angle_weight

        # Normalize the new weights to ensure they remain in a reasonable range.
        new_weights = new_weights / torch.max(new_weights)

        # Calculate the weighted sum of normals using the new weights.
        n = torch.sum(new_weights * normals, dim=1)

        # If normalization is requested, normalize the result.
        if normalize:
            n = safe_normalize(n)
        # Return the blended normals.
        return n

def angle_based_weighting(normal_a, normal_b):
    # Calculate the dot product between two normals, which is the cosine of the angle between them.
    cosine_angle = torch.clamp(torch.dot(normal_a, normal_b), -1.0, 1.0)
    # Use arccos to calculate the actual angle from the cosine value.
    angle = torch.acos(cosine_angle)
    # Assign a weight based on the angle, giving higher weights to smaller angles (more aligned normals).
    weight = torch.exp(-angle)
    # Return the calculated weight.
    return weight

