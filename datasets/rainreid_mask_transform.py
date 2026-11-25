import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from timm.data.random_erasing import RandomErasing


class _ImageWithMasks:
    """Mirror timm RandomErasing's tensor slice assignment in the masks."""

    def __init__(self, image, masks):
        self.image = image
        self.masks = masks

    @property
    def dtype(self):
        return self.image.dtype

    def size(self):
        return self.image.size()

    def __setitem__(self, index, fill):
        self.image[index] = fill
        for mask in self.masks:
            mask[index] = 0


class RainReIDMaskTransform:
    """Apply TransReID image transforms with the same geometry on both masks."""

    def __init__(self, input_cfg, training):
        self.training = training
        self.size = input_cfg.SIZE_TRAIN if training else input_cfg.SIZE_TEST
        self.interpolation = T.InterpolationMode.BICUBIC if training else T.InterpolationMode.BILINEAR
        self.flip_prob = input_cfg.PROB
        self.padding = input_cfg.PADDING
        self.normalize = T.Normalize(mean=input_cfg.PIXEL_MEAN, std=input_cfg.PIXEL_STD)
        self.erase = RandomErasing(probability=input_cfg.RE_PROB, mode='pixel', max_count=1, device='cpu')

    def __call__(self, image, masks=()):
        image = F.resize(image, self.size, interpolation=self.interpolation)
        masks = tuple(F.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST) for mask in masks)

        if self.training:
            if torch.rand(1) < self.flip_prob:
                image = F.hflip(image)
                masks = tuple(F.hflip(mask) for mask in masks)
            image = F.pad(image, self.padding)
            masks = tuple(F.pad(mask, self.padding) for mask in masks)
            crop = T.RandomCrop.get_params(image, self.size)
            image = F.crop(image, *crop)
            masks = tuple(F.crop(mask, *crop) for mask in masks)

        image = self.normalize(F.to_tensor(image))
        masks = tuple(F.to_tensor(mask) for mask in masks)
        if self.training:
            self.erase(_ImageWithMasks(image, masks) if masks else image)
        return image, masks
