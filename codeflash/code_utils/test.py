import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from inference.core.cache.model_artifacts import initialise_cache
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.env import (
    API_KEY,
    DISABLE_PREPROC_AUTO_ORIENT,
    MODEL_CACHE_DIR,
    ONNXRUNTIME_EXECUTION_PROVIDERS,
    TENSORRT_CACHE_PATH,
    USE_PYTORCH_FOR_PREPROCESSING,
)
from inference.core.models.base import Model
from inference.core.utils.image_utils import load_image
from inference.core.utils.onnx import get_onnxruntime_execution_providers
from inference.core.utils.preprocess import letterbox_image
from inference.core.utils.roboflow import get_model_id_chunks
from inference.models.aliases import resolve_roboflow_model_alias


class RoboflowInferenceModel(Model):
    def __init__(self, model_id: str, cache_dir_root=MODEL_CACHE_DIR, api_key=None, load_weights=True):
        """Initialize the RoboflowInferenceModel object.

        Args:
            model_id (str): The unique identifier for the model.
            cache_dir_root (str, optional): The root directory for the cache. Defaults to MODEL_CACHE_DIR.
            api_key (str, optional): API key for authentication. Defaults to None.

        """
        super().__init__()
        self.load_weights = load_weights
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)
        self.dataset_id, self.version_id = get_model_id_chunks(model_id=model_id)
        self.endpoint = model_id
        self.device_id = GLOBAL_DEVICE_ID
        self.cache_dir = os.path.join(cache_dir_root, self.endpoint)
        self.keypoints_metadata: Optional[dict] = None
        initialise_cache(model_id=self.endpoint)
        self.image_loader_threadpool = ThreadPoolExecutor()

    def preproc_image(
        self,
        image: Union[Any, InferenceRequestImage],
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocesses an inference request image by loading it, then applying any pre-processing specified by the Roboflow platform, then scaling it to the inference input dimensions.

        Args:
            image (Union[Any, InferenceRequestImage]): An object containing information necessary to load the image for inference.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: A tuple containing the preprocessed image pixel data and the image's original size.

        """
        np_image, is_bgr = load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient
            or "auto-orient" not in self.preproc.keys()
            or DISABLE_PREPROC_AUTO_ORIENT,
        )
        preprocessed_image, img_dims = self.preprocess_image(
            np_image,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

        if USE_PYTORCH_FOR_PREPROCESSING and torch.cuda.is_available():
            preprocessed_image = (
                torch.from_numpy(np.ascontiguousarray(preprocessed_image))
                .cuda()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
                .float()
            )

        resize_method_map = {
            "Stretch to": self._resize_stretch,
            "Fit (black edges) in": partial(letterbox_image, desired_size=(self.img_size_w, self.img_size_h)),
            "Fit (white edges) in": partial(
                letterbox_image, desired_size=(self.img_size_w, self.img_size_h), color=(255, 255, 255)
            ),
            "Fit (grey edges) in": partial(
                letterbox_image, desired_size=(self.img_size_w, self.img_size_h), color=(114, 114, 114)
            ),
        }

        resized = resize_method_map.get(self.resize_method, self._resize_stretch)(preprocessed_image)
        if is_bgr:
            if isinstance(resized, np.ndarray):
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                resized = resized[:, [2, 1, 0], :, :]

        img_in = (
            resized
            if isinstance(resized, torch.Tensor)
            else np.expand_dims(np.transpose(resized.astype(np.float32), (2, 0, 1)), axis=0)
        )
        return img_in.float() if isinstance(img_in, torch.Tensor) else img_in, img_dims


class OnnxRoboflowInferenceModel(RoboflowInferenceModel):
    def __init__(
        self,
        model_id: str,
        onnxruntime_execution_providers: List[str] = get_onnxruntime_execution_providers(
            ONNXRUNTIME_EXECUTION_PROVIDERS
        ),
        *args,
        **kwargs,
    ):
        """Initializes the OnnxRoboflowInferenceModel instance.

        Args:
            model_id (str): The identifier for the specific ONNX model.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        super().__init__(model_id, *args, **kwargs)
        if self.load_weights or not self.has_model_metadata:
            self.onnxruntime_execution_providers = onnxruntime_execution_providers or []
            self.onnxruntime_execution_providers = [
                (
                    ep
                    if ep != "TensorrtExecutionProvider"
                    else (
                        ep,
                        {
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": os.path.join(TENSORRT_CACHE_PATH, self.endpoint),
                            "trt_fp16_enable": True,
                        },
                    )
                )
                for ep in self.onnxruntime_execution_providers
            ]

    def load_image(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load image from various formats with concurrent processing.

        Args:
            image (Any): Input image data.
            disable_preproc_auto_orient (bool, optional): Disable auto-orientation. Defaults to False.
            disable_preproc_contrast (bool, optional): Disable contrast pre-processing. Defaults to False.
            disable_preproc_grayscale (bool, optional): Disable grayscale pre-processing. Defaults to False.
            disable_preproc_static_crop (bool, optional): Disable static crop pre-processing. Defaults to False.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: Loaded image and its dimensions.

        """
        if isinstance(image, list):
            preproc_func = partial(
                self.preproc_image,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                disable_preproc_contrast=disable_preproc_contrast,
                disable_preproc_grayscale=disable_preproc_grayscale,
                disable_preproc_static_crop=disable_preproc_static_crop,
            )
            imgs_with_dims = list(self.image_loader_threadpool.map(preproc_func, image))
            imgs, img_dims = zip(*imgs_with_dims)
            img_in = torch.cat(imgs, dim=0) if isinstance(imgs[0], torch.Tensor) else np.concatenate(imgs, axis=0)
        else:
            img_in, img_dims = self.preproc_image(
                image,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                disable_preproc_contrast=disable_preproc_contrast,
                disable_preproc_grayscale=disable_preproc_grayscale,
                disable_preproc_static_crop=disable_preproc_static_crop,
            )
            img_dims = [img_dims]
        return img_in, img_dims
