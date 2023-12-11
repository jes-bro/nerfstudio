import numpy as np
import sys
import torch
import json
from os import chdir
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from nerfstudio.cameras import cameras
import matplotlib.pyplot as plt
from PIL import Image
import io

def get_rgb_from_pose_transform(load_config, camera_to_world, fx= torch.Tensor([1400.]), fy=torch.Tensor([1400.]), cx=torch.Tensor([960.]), cy=torch.Tensor([960.])):
    #batch_camera_to_world = camera_to_world.unsqueeze(0)
    device = torch.device("cuda")
    camera_to_world = camera_to_world.to(device)
    load_config_path = Path(load_config)
    #breakpoint()
    config, pipeline, checkpoint_path, step = eval_setup(load_config_path)
    #print("hi")
    model = pipeline.model
    num_rays = 100
    camera_indices = torch.tensor([0], dtype=torch.int32)
    camera_indices = camera_indices.to(device)
    #camera_indices = camera_indices.unsqueeze(-1)
    #camera_indices = camera_indices.unsqueeze(0).repeat(num_rays, 1)
    camera = cameras.Cameras(camera_to_world, fx, fy, cx, cy)
    #breakpoint()
    rays = camera.generate_rays(camera_indices)
    #breakpoint()
    outputs = model.get_outputs_for_camera_ray_bundle(rays)
    #breakpoint()
    rgba_image = model.get_rgba_image(outputs)
    #print("here")
    #print(rgba_image)
    #print(len(rgba_image))
    return rgba_image

if __name__ == '__main__':
    chdir("/home/jess")
    test_path = "/home/jess/outputs/camera_pose/nerfacto/2023-12-04_221013/config.yml"
    string_c2w = sys.argv[1]
    list_c2w = json.loads(string_c2w)
    nparray_c2w = np.array([list_c2w])
    #breakpoint()
    tensor_c2w = torch.Tensor([nparray_c2w]).squeeze()
    rgba_image = get_rgb_from_pose_transform(test_path, tensor_c2w, torch.Tensor([1400.]), torch.Tensor([1400.]), torch.Tensor([960.]), torch.Tensor([960.]))
    rgba_image_cpu = rgba_image.cpu()
    rgba_numpy = rgba_image_cpu.numpy()
    plt.imshow(rgba_image_cpu)
    plt.savefig('/home/jess/ros2_ws/output_image.png', bbox_inches='tight', pad_inches=0)
    plt.axis('off') 
    #rgba_list = rgba_numpy.tolist()
    # Convert the NumPy array to an image
    #print(json_rgba_image)
    #plt.show()
    plt.close()
"""    camera_to_world = np.array([[ .5, .5, 0., 0.],[ 0., 0.25,0.,0.],[ 0.,0.,0.25,0.]])
    test_path = "/home/jess/outputs/camera_pose/nerfacto/2023-12-04_221013/config.yml"
    rgba_image = get_rgb_from_pose_transform(test_path, torch.Tensor([camera_to_world]), torch.Tensor([1400.]), torch.Tensor([1400.]), torch.Tensor([960.]), torch.Tensor([960.]))
    rgba_image_cpu = rgba_image.cpu()
    print(rgba_image_cpu)
    #breakpoint()
    plt.imshow(rgba_image_cpu)
    plt.show()
    plt.close()"""