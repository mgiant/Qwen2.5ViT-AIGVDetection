from PIL import Image
import random
import io
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms


class JpegCompressionPIL(object):
    def __init__(self, p=-1, quality=100):
        self.p = p
        if isinstance(quality, int):
            self.min_quality = quality
            self.max_quality = quality
        else:
            self.min_quality, self.max_quality = quality

    def _compress_single(self, img: Image.Image, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        compressed_image.load()

        if img.mode == 'RGBA' and compressed_image.mode != 'RGBA':
            return compressed_image.convert('RGB')
        return compressed_image

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.min_quality, self.max_quality)

            if isinstance(img, list):
                return [self._compress_single(i, quality) for i in img]
            else:
                return self._compress_single(img, quality)
        return img

class VideoRandomRotation:
    """
    Applies a random rotation to a list of images (e.g., video frames)
    consistently. This ensures that all frames in the video clip are rotated
    by the same angle.
    """
    def __init__(self, degrees):
        """
        Args:
            degrees (Union[int, float, Tuple[float, float]]): The range of degrees to rotate.
                If a number d, the range is [-d, +d].
                If a tuple (min, max), the range is [min, max].
        """
        # Pre-process the degrees argument to a standard (min, max) tuple format.
        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be non-negative.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must contain two numbers.")
            self.degrees = degrees

    def __call__(self, img_list):
        """
        Args:
            img_list (List[PIL.Image.Image] or List[torch.Tensor]):
                A list of images to be transformed.

        Returns:
            List[PIL.Image.Image] or List[torch.Tensor]:
                The list of images, all rotated by the same angle.
        """
        # Generate a single random angle for the entire video clip using NumPy.
        angle = np.random.uniform(self.degrees[0], self.degrees[1])

        # Apply this same rotation angle to every frame in the list.
        return [F.rotate(img, angle) for img in img_list]

    def __repr__(self) -> str:
        """
        Provides a clear string representation of the class for debugging.
        """
        return (f"{self.__class__.__name__}("
                f"degrees={self.degrees}, "
                f"interpolation={self.interpolation.value}")

class VideoRandomCrop:
    """
    Apply an identical random crop to a list of PIL Images or tensors.

    This transform is designed for video data and acts as a standard
    callable data augmentation step. It is not a torch.nn.Module.
    """

    def __init__(self, size):
        """
        Initializes the VideoRandomCrop transform.

        Args:
            size (int or sequence): The desired output size of the crop.
                If size is an int, a square crop of size (size, size) is made.
                If size is a sequence of (h, w), a crop of that size is made.
        """
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """
        Get parameters for a random crop. This is a static method adapted
        from torchvision's internal implementation.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop (h, w).

        Returns:
            tuple: params (i, j, h, w) to be passed to `crop` for a random crop.
        """
        w, h = F.get_image_size(img)
        th, tw = output_size

        if h < th or w < tw:
            return -1, -1, th, tw

        if w == tw and h == th:
            return 0, 0, h, w

        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return i, j, th, tw

    def __call__(self, imgs):
        """
        Apply the random crop to a list of images.

        Args:
            imgs (list of PIL Image or list of Tensor): List of frames to be cropped.

        Returns:
            list of PIL Image or list of Tensor: The cropped frames.
        """
        if not imgs:
            return imgs

        # Determine the random crop parameters once using the first frame.
        # These parameters will be applied to all frames in the list.
        crop_params = self.get_params(imgs[0], self.size)
        i, j, h, w = crop_params

        # Apply the identical crop to each frame in the list.
        if i < 0 and j < 0:
            return imgs
        return [F.crop(img, i, j, h, w) for img in imgs]

    def __repr__(self):
        """
        Provides a developer-friendly string representation of the transform.
        """
        return self.__class__.__name__ + f"(size={self.size})"

def add_gaussian_noise(image, mean=0, std=10):
    if isinstance(image, Image.Image):
        return_image = True
        image = np.array(image)

    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    if return_image:
        return Image.fromarray(noisy_image)
    else:
        return noisy_image
    
