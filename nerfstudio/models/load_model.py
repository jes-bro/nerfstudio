import numpy as np
import torch
from os import chdir
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from nerfstudio.cameras import cameras
import matplotlib.pyplot as plt

def get_rgb_from_pose_transform(load_config, camera_to_world, fx, fy, cx, cy):
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
    print("here")
    print(rgba_image)
    print(len(rgba_image))
    return rgba_image

chdir("/home/jess")
camera_to_world = np.array([[ 1., 0., 0., 3.],[ 0., 0.98078528,-0.19509032,3.],[ 0.,0.19509032,0.98078528,0.]])
test_path = "/home/jess/outputs/camera_pose/nerfacto/2023-12-04_221013/config.yml"
rgba_image = get_rgb_from_pose_transform(test_path, torch.Tensor([camera_to_world]), torch.Tensor([1400.]), torch.Tensor([1400.]), torch.Tensor([960.]), torch.Tensor([960.]))
rgba_image_cpu = rgba_image.cpu()
print(rgba_image_cpu)
breakpoint()
plt.imshow(rgba_image_cpu)
plt.show()