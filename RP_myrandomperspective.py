import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

from torch._C import _log_api_usage_once
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
#from .functional import InterpolationMode, _interpolation_modes_from_int


class MyRandomPerspective(torch.nn.Module):
    """Performs a random perspective transformation of the given image with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.
        p (float): probability of the image being transformed. Default is 0.5.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, distortion_scale=0.1, p=0.01, interpolation=InterpolationMode.BILINEAR, fill=0):
        super().__init__()
        #_log_api_usage_once(self)
        self.p = p

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. "
                "Please use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be Perspectively transformed.

        Returns:
            PIL Image or Tensor: Randomly transformed image.
        """

        fill = self.fill
        #channels, height, width = F.get_dimensions(img)
        channels = 3
        height = 512
        width = 512
        #print("CHANNELS: ", channels)
        #print("HEIGHT: ", height)
        #print("WIDTH: ", width)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        if torch.rand(1) < self.p:
            startpoints, endpoints, top, left, height_crop, width_crop = self.get_params(width, height, self.distortion_scale)
            # PROVA
           # if height_crop > 0 and width_crop > 0:
            #return F.perspective(img, startpoints, endpoints, self.interpolation, fill)
            img = F.perspective(img, startpoints, endpoints, self.interpolation, fill)
            #print("RAFFA 2: ", img.size())
            img = F.crop(img, top, left, height_crop, width_crop)
            #print("RAFFA 3: ", img.size())
            img = F.resize(img, [512, 512])
            #print("RAFFA 4: ", img.size())
            return img
            #else:
            #  return img
        return img

    @staticmethod
    def get_params(width: int, height: int, distortion_scale: float) -> Tuple[List[List[int]], List[List[int]]]:
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]

        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]

        ### RAFFA ###

        # taglio il più grande rettangolo senza pixel neri
        
        #endpoints_internal = [
        #    [max(topleft[0], botleft[0]), max(topleft[1], topright[1])],
        #    [min(topright[0], botright[0]), max(topleft[1], topright[1])],
        #    [min(topright[0], botright[0]), min(botleft[1], botright[1])],
        #    [max(topleft[0], botleft[0]), min(botleft[1], botright[1])]]
        #top =  max(topleft[1], topright[1])
        #left = max(topleft[0], botleft[0])
        #height_crop = min(botleft[1], botright[1]) - max(topleft[1], topright[1])
        #width_crop = min(topright[0], botright[0]) - max(topleft[0], botleft[0])

        #print("RAFFA INTERNO ENDPOINTS: ", endpoints_internal)

        # taglio il più piccolo rettangolo senza perdere pixel colorati

        #endpoints_external = [
        #    [min(topleft[0], botleft[0]), min(topleft[1], topright[1])],
        #    [max(topright[0], botright[0]), min(topleft[1], topright[1])],
        #    [max(topright[0], botright[0]), max(botleft[1], botright[1])],
        #    [min(topleft[0], botleft[0]), max(botleft[1], botright[1])]]
        top =  min(topleft[1], topright[1])
        left = min(topleft[0], botleft[0])
        height_crop = max(botleft[1], botright[1]) - min(topleft[1], topright[1])
        width_crop = max(topright[0], botright[0]) - min(topleft[0], botleft[0])

        #print("RAFFA ESTERNO ENDPOINTS: ", endpoints)

        # in realtà facendo così non prendo necessariamente il rettangolo più grande,
        # quindi provo a fare una media invece di max/min, per trovare un trade off
        # (diminuisco i pixel neri senza rimuoverli del tutto, ma aumento i pixel colorati)

        #endpoints = [
        #    [0.5*(endpoints_internal[0][0]+endpoints_external[0][0]), 0.5*(endpoints_internal[0][1]+endpoints_external[0][1])],
        #    [0.5*(endpoints_internal[1][0]+endpoints_external[1][0]), 0.5*(endpoints_internal[1][1]+endpoints_external[1][1])],
        #    [0.5*(endpoints_internal[2][0]+endpoints_external[2][0]), 0.5*(endpoints_internal[2][1]+endpoints_external[2][1])],
        #    [0.5*(endpoints_internal[3][0]+endpoints_external[3][0]), 0.5*(endpoints_internal[3][1]+endpoints_external[3][1])]]
        #top =  0.5*(max(topleft[1], topright[1]) + min(topleft[1], topright[1]))
        #left = 0.5*(max(topleft[0], botleft[0]) + min(topleft[0], botleft[0]))
        #height_crop = 0.5*(min(botleft[1], botright[1]) - max(topleft[1], topright[1]))\
        #                    - 0.5*(max(botleft[1], botright[1]) - min(topleft[1], topright[1]))
        #width_crop = 0.5*(min(topright[0], botright[0]) + max(topright[0], botright[0]))\
        #                - 0.5*(max(botleft[0], topleft[0]) + min(botleft[0], topleft[0]))
        
        #endpoints = [topleft, topright, botright, botleft]
        #print("REAL ENDPOINTS: ", endpoints)

        return startpoints, endpoints, top, left, height_crop, width_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
