import numpy as np
import sys
import torch
import json
from os import chdir
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from nerfstudio.cameras import cameras
import matplotlib.pyplot as plt

def get_rgb_from_pose_transform(load_config, camera_to_world, fx= torch.Tensor([1400.]), fy=torch.Tensor([1400.]), cx=torch.Tensor([960.]), cy=torch.Tensor([960.])):
    # Specify that forward pass should be on GPU
    device = torch.device("cuda")
    # Make sure c2w matrix is stored on GPU
    camera_to_world = camera_to_world.to(device)
    # Create a path object out of the config so eval_setup can use it as an input
    load_config_path = Path(load_config)
    # Get pipeline frome eval_setup. We get pipeline so we can extract the model
    config, pipeline, checkpoint_path, step = eval_setup(load_config_path)
    # Extract the model from the pipeline
    model = pipeline.model
    # Specify that the forward pass is from one cameras worth of pose data (I don't even know what it would mean to have more)
    camera_indices = torch.tensor([0], dtype=torch.int32)
    # Port that tensor to the GPU
    camera_indices = camera_indices.to(device)
    # Create a cameras object
    camera = cameras.Cameras(camera_to_world, fx, fy, cx, cy)
    # Generate rays from the cameras object- this is the actual input to the NeRF model
    rays = camera.generate_rays(camera_indices)
    # Pass rays into model to get NeRF output
    outputs = model.get_outputs_for_camera_ray_bundle(rays)
    # Get the rgba image from the NeRF output
    rgba_image = model.get_rgba_image(outputs)
    # Return it
    return rgba_image

if __name__ == '__main__':
    chdir("/home/jess")
    test_path = "/home/jess/outputs/poly_data/nerfacto/2023-12-16_041906" #"/home/jess/outputs/home/jess/outputs/new_stuff/nerfacto/2023-12-16_024242/config.yml"
    # Parse the CL arguments to extract the camera to world pose transform
    string_c2w = sys.argv[1]
    # Load with json (as Python list)
    list_c2w = json.loads(string_c2w)
    # Convert to numpy array
    nparray_c2w = np.array([list_c2w])
    # Convert to tensor
    tensor_c2w = torch.Tensor([nparray_c2w]).squeeze()
    # Get RGBA image
    rgba_image = get_rgb_from_pose_transform(test_path, tensor_c2w, torch.Tensor([1400.]), torch.Tensor([1400.]), torch.Tensor([960.]), torch.Tensor([960.]))
    # Port it to CPU
    rgba_image_cpu = rgba_image.cpu()
    # Convert to numy arrray again
    rgba_numpy = rgba_image_cpu.numpy()
    # Show the image to verify it came through
    plt.imshow(rgba_image_cpu)
    # Save the image to a file
    plt.savefig('/home/jess/ros2_ws/output_image.png', bbox_inches='tight', pad_inches=0)
    # Turn off axes so they don't show up in the picture
    plt.axis('off') 
    # Close the plot cleanly
    plt.close()