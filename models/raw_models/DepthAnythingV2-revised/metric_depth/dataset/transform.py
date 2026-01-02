import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        if np.isnan(x) or np.isinf(x):
            raise ValueError(f"Invalid input to constrain_to_multiple_of: x={x} (NaN or Inf)")
        
        if x < 0:
            raise ValueError(f"Invalid input to constrain_to_multiple_of: x={x} (negative)")
        
        # Calculate the constrained value
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)
            # Ensure it doesn't exceed max_val
            if y > max_val:
                y = ((max_val // self.__multiple_of) * self.__multiple_of)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)
            # Ensure it meets min_val
            if y < min_val:
                y = ((min_val + self.__multiple_of - 1) // self.__multiple_of) * self.__multiple_of
        
        # Final validation - ensure we never return 0 or negative
        if y <= 0:
            # Use the maximum of multiple_of and min_val, but ensure it's at least multiple_of
            y = max(self.__multiple_of, min_val) if min_val > 0 else self.__multiple_of
            # Ensure it's still a multiple
            y = ((y + self.__multiple_of - 1) // self.__multiple_of) * self.__multiple_of
        
        # Final check for validity
        if y <= 0 or np.isnan(y) or np.isinf(y):
            raise ValueError(
                f"constrain_to_multiple_of returned invalid value: {y} "
                f"(input: x={x}, min_val={min_val}, max_val={max_val}, multiple_of={self.__multiple_of})"
            )

        return int(y)

    def get_size(self, width, height):
        # Validate input dimensions
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid input dimensions: width={width}, height={height}. "
                f"Both must be positive integers."
            )
        
        if self.__width <= 0 or self.__height <= 0:
            raise ValueError(
                f"Invalid target dimensions: width={self.__width}, height={self.__height}. "
                f"Both must be positive integers."
            )
        
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width
        
        # Validate scales
        if np.isnan(scale_height) or np.isnan(scale_width) or np.isinf(scale_height) or np.isinf(scale_width):
            raise ValueError(
                f"Invalid scale calculated: scale_width={scale_width}, scale_height={scale_height}. "
                f"Input: {width}x{height}, Target: {self.__width}x{self.__height}"
            )

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=1)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=1)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")
        
        # Final validation of calculated dimensions
        if new_width <= 0 or new_height <= 0:
            raise ValueError(
                f"get_size() returned invalid dimensions: width={new_width}, height={new_height}. "
                f"Input: {width}x{height}, Target: {self.__width}x{self.__height}, "
                f"Scales: {scale_width}x{scale_height}, Method: {self.__resize_method}"
            )
        
        if np.isnan(new_width) or np.isnan(new_height) or np.isinf(new_width) or np.isinf(new_height):
            raise ValueError(
                f"get_size() returned NaN/Inf dimensions: width={new_width}, height={new_height}. "
                f"Input: {width}x{height}, Target: {self.__width}x{self.__height}"
            )

        return (new_width, new_height)

    def __call__(self, sample):
        # Validate input image dimensions
        if "image" not in sample:
            raise ValueError("Sample must contain 'image' key")
        
        image = sample["image"]
        if len(image.shape) < 2:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected at least 2 dimensions (H, W) or (H, W, C)")
        
        # Get image dimensions (handle both (H, W) and (H, W, C) formats)
        if len(image.shape) == 2:
            img_height, img_width = image.shape
        else:
            img_height, img_width = image.shape[0], image.shape[1]
        
        # Validate input dimensions
        if img_width <= 0 or img_height <= 0:
            raise ValueError(
                f"Invalid image dimensions: width={img_width}, height={img_height}. "
                f"Image shape: {image.shape}. This usually indicates a corrupted or empty image file."
            )
        
        # Get target size
        width, height = self.get_size(img_width, img_height)
        
        # Validate output dimensions
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid output dimensions calculated: width={width}, height={height}. "
                f"Input dimensions: width={img_width}, height={img_height}. "
                f"Target size: width={self.__width}, height={self.__height}. "
                f"This may indicate an issue with the resize parameters or input image."
            )
        
        # Ensure dimensions are integers and validate
        try:
            width = int(width)
            height = int(height)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert dimensions to integers: width={width} (type: {type(width)}), "
                f"height={height} (type: {type(height)}). Original error: {e}"
            )
        
        # Check for NaN, Inf, or invalid values
        if np.isnan(width) or np.isnan(height) or np.isinf(width) or np.isinf(height):
            raise ValueError(
                f"Invalid dimensions (NaN or Inf): width={width}, height={height}. "
                f"Input dimensions: width={img_width}, height={img_height}. "
                f"Target size: width={self.__width}, height={self.__height}."
            )
        
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Output dimensions must be positive integers, got: width={width}, height={height}. "
                f"Input dimensions: width={img_width}, height={img_height}. "
                f"Target size: width={self.__width}, height={self.__height}."
            )
        
        # Check for unreasonably large dimensions (likely a calculation error)
        if width > 100000 or height > 100000:
            raise ValueError(
                f"Unreasonably large dimensions calculated: width={width}, height={height}. "
                f"This likely indicates a calculation error. Input: {img_width}x{img_height}, "
                f"Target: {self.__width}x{self.__height}."
            )

        # resize sample with error handling
        try:
            dsize = (int(width), int(height))
            # Double-check the tuple is valid
            if len(dsize) != 2 or dsize[0] <= 0 or dsize[1] <= 0:
                raise ValueError(f"Invalid dsize tuple: {dsize}")
            
            sample["image"] = cv2.resize(
                sample["image"],
                dsize,
                interpolation=self.__image_interpolation_method,
            )
        except cv2.error as e:
            raise ValueError(
                f"cv2.resize failed with dimensions: width={width}, height={height}, dsize={dsize}. "
                f"Input image shape: {image.shape}, Input dimensions: {img_width}x{img_height}, "
                f"Target size: {self.__width}x{self.__height}. "
                f"OpenCV error: {e}"
            ) from e

        if self.__resize_target:
            dsize = (int(width), int(height))
            
            if "disparity" in sample:
                try:
                    sample["disparity"] = cv2.resize(
                        sample["disparity"],
                        dsize,
                        interpolation=cv2.INTER_NEAREST,
                    )
                except cv2.error as e:
                    raise ValueError(
                        f"cv2.resize failed for disparity with dsize={dsize}. "
                        f"Disparity shape: {sample['disparity'].shape}. Error: {e}"
                    ) from e

            if "depth" in sample:
                try:
                    sample["depth"] = cv2.resize(
                        sample["depth"], dsize, interpolation=cv2.INTER_NEAREST
                    )
                except cv2.error as e:
                    raise ValueError(
                        f"cv2.resize failed for depth with dsize={dsize}. "
                        f"Depth shape: {sample['depth'].shape}. Error: {e}"
                    ) from e

            if "semseg_mask" in sample:
                # sample["semseg_mask"] = cv2.resize(
                #     sample["semseg_mask"], (width, height), interpolation=cv2.INTER_NEAREST
                # )
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode='nearest').numpy()[0, 0]
                
            if "mask" in sample:
                try:
                    sample["mask"] = cv2.resize(
                        sample["mask"].astype(np.float32),
                        dsize,
                        interpolation=cv2.INTER_NEAREST,
                    )
                except cv2.error as e:
                    raise ValueError(
                        f"cv2.resize failed for mask with dsize={dsize}. "
                        f"Mask shape: {sample['mask'].shape}. Error: {e}"
                    ) from e
                # sample["mask"] = sample["mask"].astype(bool)

        # print(sample['image'].shape, sample['depth'].shape)
        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        
        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
            
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample


class Crop(object):
    """Crop sample for batch-wise training. Image is of shape CxHxW
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample['image'].shape[-2:]
        assert h >= self.size[0] and w >= self.size[1], 'Wrong size'
        
        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        h_end = h_start + self.size[0]
        w_end = w_start + self.size[1]
        
        sample['image'] = sample['image'][:, h_start: h_end, w_start: w_end]
        
        if "depth" in sample:
            sample["depth"] = sample["depth"][h_start: h_end, w_start: w_end]
        
        if "mask" in sample:
            sample["mask"] = sample["mask"][h_start: h_end, w_start: w_end]
            
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][h_start: h_end, w_start: w_end]
            
        return sample