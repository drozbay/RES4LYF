import re
import torch
from comfy.samplers import SCHEDULER_NAMES
from comfy import model_sampling
import torch.nn.functional as F

def filter_comments(extra_options):
    return "\n".join(line for line in extra_options.splitlines() if not line.strip().startswith("#"))

def get_extra_options_kv(key, default, extra_options):
    extra_options = filter_comments(extra_options)
    match = re.search(rf"{key}\s*=\s*([a-zA-Z0-9_.+-]+)", extra_options)
    if match:
        value = match.group(1)
    else:
        value = default
    return value

def get_extra_options_list(key, default, extra_options):

    match = re.search(rf"{key}\s*=\s*([a-zA-Z0-9_.,+-]+)", extra_options)
    if match:
        value = match.group(1)
    else:
        value = default
    return value

def extra_options_flag(flag, extra_options):
    extra_options = filter_comments(extra_options)
    return bool(re.search(rf"{flag}", extra_options))


def safe_get_nested(d, keys, default=None):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

def is_video_model(model):
    is_video_model = False
    if type(model) is dict:
        if hasattr(model, 'inner_model') and \
            hasattr(model.inner_model, 'inner_model') and \
            hasattr(model.inner_model.inner_model, 'model_config') and \
            hasattr(model.inner_model.inner_model.model_config, 'unet_config'):
                if 'image_model' in model.inner_model.inner_model.model_config.unet_config:
                    is_video_model = \
                    'video' in model.inner_model.inner_model.model_config.unet_config['image_model'] or \
                    'cosmos' in model.inner_model.inner_model.model_config.unet_config['image_model']
    return is_video_model

def is_RF_model(model) -> bool:
    modelsampling = model.inner_model.inner_model.model_sampling
    return isinstance(modelsampling, model_sampling.CONST)

def get_cosine_similarity_manual(a, b) -> torch.Tensor:
    return (a * b).sum() / (torch.norm(a) * torch.norm(b))



def get_cosine_similarity(a, b) -> torch.Tensor:
    if a.dim() == 5 and b.dim() == 5 and b.shape[2] == 1:
        b = b.expand(-1, -1, a.shape[2], -1, -1)
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)


def get_pearson_similarity(a, b):
    a = a.mean(dim=(-2,-1))
    b = b.mean(dim=(-2,-1))
    if a.dim() == 5 and b.dim() == 5 and b.shape[2] == 1:
        b = b.expand(-1, -1, a.shape[2], -1, -1)
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)



def initialize_or_scale(tensor, value, steps) -> torch.Tensor:
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor


def has_nested_attr(obj, attr_path) -> bool:
    attrs = attr_path.split('.')
    for attr in attrs:
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return True

def get_res4lyf_scheduler_list() -> list:
    scheduler_names = SCHEDULER_NAMES.copy()
    if "beta57" not in scheduler_names:
        scheduler_names.append("beta57")
    return scheduler_names

def conditioning_set_values(conditioning, values={}) -> list:
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c



# pytorch slerp implementation from https://gist.github.com/Birch-san/230ac46f99ec411ed5907b0a3d728efa
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm

# adapted to PyTorch from:
# https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
# most of the extra complexity is to support:
# - many-dimensional vectors
# - v0 or v1 with last dim all zeroes, or v0 ~colinear with v1
#   - falls back to lerp()
#   - conditional logic implemented with parallelism rather than Python loops
# - many-dimensional tensor for t
#   - you can ask for batches of slerp outputs by making t more-dimensional than the vectors
#   -   slerp(
#         v0:   torch.Size([2,3]),
#         v1:   torch.Size([2,3]),
#         t:  torch.Size([4,1,1]), 
#       )
#   - this makes it interface-compatible with lerp()
def slerp(v0: FloatTensor, v1: FloatTensor, t: float|FloatTensor, DOT_THRESHOLD=0.9995) -> FloatTensor:
  '''
  Spherical linear interpolation
  Args:
    v0: Starting vector
    v1: Final vector
    t: Float value between 0.0 and 1.0
    DOT_THRESHOLD: Threshold for considering the two vectors as
                            colinear. Not recommended to alter this.
  Returns:
      Interpolation vector between v0 and v1
  '''
  assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

  # Normalize the vectors to get the directions and angles
  v0_norm: FloatTensor = norm(v0, dim=-1)
  v1_norm: FloatTensor = norm(v1, dim=-1)

  v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
  v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

  # Dot product with the normalized vectors
  dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
  dot_mag: FloatTensor = dot.abs()

  # if dp is NaN, it's because the v0 or v1 row was filled with 0s
  # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
  gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
  can_slerp: LongTensor = ~gotta_lerp

  t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, Tensor) else 0
  t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
  out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

  # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
  if gotta_lerp.any():
    lerped: FloatTensor = lerp(v0, v1, t)

    out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

  # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
  if can_slerp.any():

    # Calculate initial angle between v0 and v1
    theta_0: FloatTensor = dot.arccos().unsqueeze(-1)
    sin_theta_0: FloatTensor = theta_0.sin()
    # Angle at timestep t
    theta_t: FloatTensor = theta_0 * t
    sin_theta_t: FloatTensor = theta_t.sin()
    # Finish the slerp algorithm
    s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
    s1: FloatTensor = sin_theta_t / sin_theta_0
    slerped: FloatTensor = s0 * v0 + s1 * v1

    out: FloatTensor = slerped.where(can_slerp.unsqueeze(-1), out)
  
  return out