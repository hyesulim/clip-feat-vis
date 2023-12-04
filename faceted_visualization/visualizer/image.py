from typing import List

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2

# based on code from https://github.com/greentfrapp/lucent/tree/dev/lucent/optvis/param

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]).astype(
    "float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt


def _linear_decorrelate_color(tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_permute = tensor.permute(0, 2, 3, 1)
    t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(device))
    tensor = t_permute.permute(0, 3, 1, 2)
    return tensor


def to_valid_rgb(image_f, decorrelate=False):
    def inner():
        image = image_f()
        if decorrelate:
            image = _linear_decorrelate_color(image)
        return torch.sigmoid(image)

    return inner


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape, sd=None, decay_power=1, device="cpu"):
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, channels) + freqs.shape + (2,)  # 2 for imaginary and real components
    sd = sd or 0.01

    spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if True:
            import torch.fft
            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        else:
            import torch
            image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        image = image[:batch, :channels, :h, :w]
        magic = 4.0  # Magic constant from Lucid library; increasing this seems to reduce saturation
        image = image / magic
        return image

    return [spectrum_real_imag_t], inner


def pixel_image(shape, sd=None, device="cpu"):
    sd = sd or 0.01
    tensor = (torch.randn(*shape) * sd).to(device).requires_grad_(True)
    return [tensor], lambda: tensor


def generate_img(w, h=None, sd=None, batch=None, decorrelate=True, fft=True, channels=None, device="cpu"):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd, device=device)
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def convert_to_PIL(tensor) -> Image.Image:
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    return Image.fromarray(image)


def consolidate_transforms(use_clip_transforms: bool, use_standard_transforms: bool,
                           clip_transforms: torchvision.transforms.Compose):
    if not use_clip_transforms:
        clip_transforms = []

    if use_standard_transforms:
        standard_transforms = get_standard_transforms()
    else:
        standard_transforms = []
    transforms = clip_transforms + standard_transforms
    if len(transforms) > 0:
        transforms = torchvision.transforms.v2.Compose(transforms=transforms)
    else:
        transforms = None
    return transforms


def get_standard_transforms() -> List:
    return [
        torchvision.transforms.v2.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        torchvision.transforms.v2.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        torchvision.transforms.v2.RandomRotation((-10, 10)),
        torchvision.transforms.v2.RandomAffine(degrees=0, translate=(0.015, 0.015)),
    ]