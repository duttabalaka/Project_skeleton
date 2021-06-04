# Project_skeleton

## Run demo:

```bash
PYTHONPATH="./" bin/main --image_path ./docs/sample_input.jpg  --image_truth_path ./docs/sample_gt.png  --max_radius 10 --show --device cuda
```


## Get a radial skeleton
```python
import sys;sys.path.append("./")
import skeleton_stabilize
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt

device="cpu"
max_radius=10
bin_img=torch.Tensor(np.array(Image.open("./docs/sample_gt.png")),device=device)
filters=skeleton_stabilize.create_circular_filter(max_radius, min_radius=0, max_radius=(max_radius-1), device=device)
radial_skeleton = skeleton_stabilize.extract_radial_skeleton(bin_img)

# play with radial skeleton
# and then render the result

stable_bin_img = skeleton_stabilize.render_radial_skeleton(radial_skeleton=radial_skeleton, filter=filters,device=device)

plt.imshow(torch.cat([bin_img, stable_bin_img], dim=1).numpy())
plt.show()



```
