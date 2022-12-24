from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from torch.utils.data import Dataset

from monai.config import DtypeLike
from monai.data.image_reader import ImageReader
from monai.transforms import LoadImage, Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed

class TwoImageDataset(Dataset, Randomizable):
    """
    Loads image/segmentation pairs of files from the given filename lists. Transformations can be specified
    for the image and segmentation arrays separately.
    The difference between this dataset and `ArrayDataset` is that this dataset can apply transform chain to images
    and segs and return both the images and metadata, and no need to specify transform to load images from files.
    For more information, please see the image_dataset demo in the MONAI tutorial repo,
    https://github.com/Project-MONAI/tutorials/blob/master/modules/image_dataset.ipynb
    """
    def __init__(
        self,
        image_files: Sequence[str],
        seg_files: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[float]] = None,
        transform: Optional[Callable] = None,
        seg_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        image_only: bool = True,
        transform_with_metadata: bool = False,
        dtype: DtypeLike = np.float32,
        reader: Optional[Union[ImageReader, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the dataset with the image and segmentation filename lists. The transform `transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            image_files: list of image filenames.
            seg_files: if in segmentation task, list of segmentation filenames.
            labels: if in classification task, list of classification labels.
            transform: transform to apply to image arrays.
            seg_transform: transform to apply to segmentation arrays.
            label_transform: transform to apply to the label data.
            image_only: if True return only the image volume, otherwise, return image volume and the metadata.
            transform_with_metadata: if True, the metadata will be passed to the transforms whenever possible.
            dtype: if not None convert the loaded image to this data type.
            reader: register reader to load image file and metadata, if None, will use the default readers.
                If a string of reader name provided, will construct a reader object with the `*args` and `**kwargs`
                parameters, supported reader name: "NibabelReader", "PILReader", "ITKReader", "NumpyReader"
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.

        Raises:
            ValueError: When ``seg_files`` length differs from ``image_files``

        """

        if seg_files is not None and len(image_files) != len(seg_files):
            raise ValueError(
                "Must have same the number of segmentation as image files: "
                f"images={len(image_files)}, segmentations={len(seg_files)}."
            )

        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.transform = transform
        self.seg_transform = seg_transform
        self.label_transform = label_transform
        if image_only and transform_with_metadata:
            raise ValueError("transform_with_metadata=True requires image_only=False.")
        self.image_only = image_only
        self.transform_with_metadata = transform_with_metadata
        self.loader = LoadImage(reader, image_only, dtype, *args, **kwargs)
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed


    def __len__(self) -> int:
        return len(self.image_files)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")


    def __getitem__(self, index: int):
        self.randomize()
        meta_data, seg_meta_data, seg, label = None, None, None, None

        # load data and optionally meta
        if self.image_only:
            img1= self.loader(self.image_files[index][0])
            img2 = self.loader(self.image_files[index][1])
            # if self.seg_files is not None:
            #     seg = self.loader(self.seg_files[index])
        else:
            img1, meta_data1 = self.loader(self.image_files[index][0])
            img2, meta_data2 = self.loader(self.image_files[index][1])
            # if self.seg_files is not None:
            #     seg, seg_meta_data = self.loader(self.seg_files[index])

        # apply the transforms
        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                img1, meta_data1 = apply_transform(self.transform, (img1, meta_data1), map_items=False, unpack_items=True)
                img2, meta_data2 = apply_transform(self.transform, (img2, meta_data2), map_items=False, unpack_items=True)
            else:
                img1 = apply_transform(self.transform, img1, map_items=False)
                img2 = apply_transform(self.transform, img2, map_items=False)

        # if self.seg_files is not None and self.seg_transform is not None:
        #     if isinstance(self.seg_transform, Randomizable):
        #         self.seg_transform.set_random_state(seed=self._seed)

        #     if self.transform_with_metadata:
        #         seg, seg_meta_data = apply_transform(
        #             self.seg_transform, (seg, seg_meta_data), map_items=False, unpack_items=True
        #         )
        #     else:
        #         seg = apply_transform(self.seg_transform, seg, map_items=False)

        if self.labels is not None:
            print(index)
            print(len(self.labels))
            label = self.labels[index]
            if self.label_transform is not None:
                label = apply_transform(self.label_transform, label, map_items=False)  # type: ignore

        # construct outputs
        data = np.stack((np.squeeze(img1, axis=0), np.squeeze(img2, axis=0)))
        data=[data,label]
        # if seg is not None:
        #     data.append(seg)
        # if label is not None:
        #     data.append(label)
        # if not self.image_only and meta_data is not None:
        #     data.append(meta_data1, meta_data2)
        # # if not self.image_only and seg_meta_data is not None:
        # #     data.append(seg_meta_data)
        if len(data) == 1:
            return data[0]
        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        print(len(data))
        return tuple(data)
