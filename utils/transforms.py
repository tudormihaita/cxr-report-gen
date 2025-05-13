from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class TransformBuilder:
    def __init__(self, image_size=224, min_scale=0.8, normalize_stats=None, interpolation=InterpolationMode.BICUBIC):
        self.image_size = image_size
        self.min_scale = min_scale
        self.normalize_stats = normalize_stats or ((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
        self.interpolation = interpolation

    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def build(self, split):
        normalize = transforms.Normalize(self.normalize_stats[0], self.normalize_stats[1])

        if split == 'train':
            return transforms.Compose([
                transforms.Resize(self.image_size, interpolation=self.interpolation),
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(self.min_scale, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                self._convert_image_to_rgb,
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
                self._convert_image_to_rgb,
                transforms.ToTensor(),
                normalize
            ])
