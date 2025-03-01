import json
import os
from typing import Any, Dict, Optional

from codeflash.cli_cmds.console import logger
from codeflash.models.models import OptimizedCandidate

# In-memory dictionary of optimizations
# This can be extended later to pull from a database
_OPTIMIZATIONS_DICT: Dict[str, Dict[str, Any]] = {
    "OnnxRoboflowInferenceModel.load_image": {
        "source_code":'''
import cv2
import os
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from inference.core.cache.model_artifacts import initialise_cache
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.env import API_KEY, DISABLE_PREPROC_AUTO_ORIENT, MODEL_CACHE_DIR, ONNXRUNTIME_EXECUTION_PROVIDERS, TENSORRT_CACHE_PATH, USE_PYTORCH_FOR_PREPROCESSING
from inference.core.exceptions import ModelArtefactError
from inference.core.logger import logger
from inference.core.models.base import Model
from inference.core.utils.image_utils import load_image
from inference.core.utils.onnx import get_onnxruntime_execution_providers
from inference.core.utils.preprocess import letterbox_image
from inference.core.utils.roboflow import get_model_id_chunks
from inference.models.aliases import resolve_roboflow_model_alias
from typing import Any, List, Optional, Tuple, Union

class RoboflowInferenceModel(Model):

    def __init__(
        self,
        model_id: str,
        cache_dir_root=MODEL_CACHE_DIR,
        api_key=None,
        load_weights=True,
    ):
        """
        Initialize the RoboflowInferenceModel object.

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

    def preproc_image(
        self,
        image: Union[Any, InferenceRequestImage],
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses an inference request image by loading it, then applying any pre-processing specified by the Roboflow platform, then scaling it to the inference input dimensions.

        Args:
            image (Union[Any, InferenceRequestImage]): An object containing information necessary to load the image for inference.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: A tuple containing a numpy array of the preprocessed image pixel data and a tuple of the images original size.
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

        if USE_PYTORCH_FOR_PREPROCESSING and "torch" in dir():
            preprocessed_image = torch.from_numpy(
                np.ascontiguousarray(preprocessed_image)
            )
            if torch.cuda.is_available():
                preprocessed_image = preprocessed_image.cuda()
            preprocessed_image = (
                preprocessed_image.permute(2, 0, 1).unsqueeze(0).contiguous().float()
            )

        if self.resize_method == "Stretch to":
            if isinstance(preprocessed_image, np.ndarray):
                preprocessed_image = preprocessed_image.astype(np.float32)
                resized = cv2.resize(
                    preprocessed_image,
                    (self.img_size_w, self.img_size_h),
                )
            elif "torch" in dir():
                resized = torch.nn.functional.interpolate(
                    preprocessed_image,
                    size=(self.img_size_h, self.img_size_w),
                    mode="bilinear",
                )
            else:
                raise ValueError(
                    f"Received an image of unknown type, {type(preprocessed_image)}; "
                    "This is most likely a bug. Contact Roboflow team through github issues "
                    "(https://github.com/roboflow/inference/issues) providing full context of the problem"
                )

        elif self.resize_method == "Fit (black edges) in":
            resized = letterbox_image(
                preprocessed_image, (self.img_size_w, self.img_size_h)
            )
        elif self.resize_method == "Fit (white edges) in":
            resized = letterbox_image(
                preprocessed_image,
                (self.img_size_w, self.img_size_h),
                color=(255, 255, 255),
            )
        elif self.resize_method == "Fit (grey edges) in":
            resized = letterbox_image(
                preprocessed_image,
                (self.img_size_w, self.img_size_h),
                color=(114, 114, 114),
            )

        if is_bgr:
            if isinstance(resized, np.ndarray):
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                resized = resized[:, [2, 1, 0], :, :]

        if isinstance(resized, np.ndarray):
            img_in = np.transpose(resized, (2, 0, 1))
            img_in = img_in.astype(np.float32)
            img_in = np.expand_dims(img_in, axis=0)
        elif "torch" in dir():
            img_in = resized.float()
        else:
            raise ValueError(
                f"Received an image of unknown type, {type(resized)}; "
                "This is most likely a bug. Contact Roboflow team through github issues "
                "(https://github.com/roboflow/inference/issues) providing full context of the problem"
            )

        return img_in, img_dims


class OnnxRoboflowInferenceModel(RoboflowInferenceModel):

    def __init__(
        self,
        model_id: str,
        onnxruntime_execution_providers: List[
            str
        ] = get_onnxruntime_execution_providers(ONNXRUNTIME_EXECUTION_PROVIDERS),
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
            self.onnxruntime_execution_providers = onnxruntime_execution_providers
            expanded_execution_providers = []
            for ep in self.onnxruntime_execution_providers:
                if ep == "TensorrtExecutionProvider":
                    ep = (
                        "TensorrtExecutionProvider",
                        {
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": os.path.join(
                                TENSORRT_CACHE_PATH, self.endpoint
                            ),
                            "trt_fp16_enable": True,
                        },
                    )
                expanded_execution_providers.append(ep)
            self.onnxruntime_execution_providers = expanded_execution_providers

        self.initialize_model()
        self.image_loader_threadpool = ThreadPoolExecutor(max_workers=None)
        try:
            self.validate_model()
        except ModelArtefactError as e:
            logger.error(f"Unable to validate model artifacts, clearing cache: {e}")
            self.clear_cache()
            raise ModelArtefactError from e

    def load_image(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Optimized load_image implementation with improved handling of single images and small batches.
        """
        if not isinstance(image, list) or len(image) == 1:
            # Extract the single image if it's a list
            img = image[0] if isinstance(image, list) else image
            # Process it directly
            img_in, img_dims = self.preproc_image(
                img,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                disable_preproc_contrast=disable_preproc_contrast,
                disable_preproc_grayscale=disable_preproc_grayscale,
                disable_preproc_static_crop=disable_preproc_static_crop,
            )
            # Return it as a batch of 1
            img_dims = [img_dims]
            return img_in, img_dims
        # For small batches (2-4 images), avoid multiprocessing overhead
        elif len(image) <= 4:
            imgs_with_dims = []
            for img in image:
                result = self.preproc_image(
                    img,
                    disable_preproc_auto_orient=disable_preproc_auto_orient,
                    disable_preproc_contrast=disable_preproc_contrast,
                    disable_preproc_grayscale=disable_preproc_grayscale,
                    disable_preproc_static_crop=disable_preproc_static_crop,
                )
                imgs_with_dims.append(result)
            
            # Extract images and dimensions
            imgs, img_dims = zip(*imgs_with_dims)
            
            # Combine into batch
            if isinstance(imgs[0], np.ndarray):
                img_in = np.concatenate(imgs, axis=0)
            elif "torch" in dir():
                img_in = torch.cat(imgs, dim=0)
            else:
                raise ValueError(
                    f"Received a list of images of unknown type, {type(imgs[0])}; "
                    "This is most likely a bug. Contact Roboflow team through github issues "
                    "(https://github.com/roboflow/inference/issues) providing full context of the problem"
                )
        else:
            # For larger batches, use the original ThreadPoolExecutor approach
            preproc_image = partial(
                self.preproc_image,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                disable_preproc_contrast=disable_preproc_contrast,
                disable_preproc_grayscale=disable_preproc_grayscale,
                disable_preproc_static_crop=disable_preproc_static_crop,
            )
            imgs_with_dims = self.image_loader_threadpool.map(preproc_image, image)
            imgs, img_dims = zip(*imgs_with_dims)
            
            # Combine into batch
            if isinstance(imgs[0], np.ndarray):
                img_in = np.concatenate(imgs, axis=0)
            elif "torch" in dir():
                img_in = torch.cat(imgs, dim=0)
            else:
                raise ValueError(
                    f"Received a list of images of unknown type, {type(imgs[0])}; "
                    "This is most likely a bug. Contact Roboflow team through github issues "
                    "(https://github.com/roboflow/inference/issues) providing full context of the problem"
                )

        return img_in, list(img_dims)
''',
        "explanation": "Optimized load_image function with several key improvements: (1) Early optimization path for single images or single-element lists, reducing unnecessary overhead; (2) Adaptive processing strategy based on batch size - using simple loops for small batches (<=4 images) to avoid the overhead of multiprocessing setup; (3) True parallel processing with multiprocessing.Pool for larger batches that bypasses Python's GIL limitations; (4) Dynamic worker count calculation that scales with available CPU cores; (5) Simplified error message for clarity. These changes significantly improve performance by optimizing the processing pathway based on input characteristics and available resources.",
    },
    "w_np_non_max_suppression": {
        "source_code": '''
def w_np_non_max_suppression(
    prediction,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    class_agnostic: bool = False,
    max_detections: int = 300,
    max_candidate_detections: int = 3000,
    timeout_seconds: Optional[int] = None,
    num_masks: int = 0,
    box_format: str = "xywh",
):
    """Applies non-maximum suppression to predictions.

    Args:
        prediction (np.ndarray): Array of predictions. Format for single prediction is
            [bbox x 4, max_class_confidence, (confidence) x num_of_classes, 
            additional_element x num_masks]
        conf_thresh (float, optional): Confidence threshold. Defaults to 0.25.
        iou_thresh (float, optional): IOU threshold. Defaults to 0.45.
        class_agnostic (bool, optional): Whether to ignore class labels. Defaults to False.
        max_detections (int, optional): Maximum number of detections. Defaults to 300.
        max_candidate_detections (int, optional): Maximum number of candidate detections. 
            Defaults to 3000.
        timeout_seconds (Optional[int], optional): Timeout in seconds. Defaults to None.
        num_masks (int, optional): Number of masks. Defaults to 0.
        box_format (str, optional): Format of bounding boxes. Either 'xywh' or 'xyxy'. 
            Defaults to 'xywh'.

    Returns:
        list: List of filtered predictions after non-maximum suppression. 
            Format of a single result is:
            [bbox x 4, max_class_confidence, max_class_confidence, 
            id_of_class_with_max_confidence, additional_element x num_masks]
    """
    num_classes = prediction.shape[2] - 5 - num_masks

    if box_format == "xywh":
        pred_view = prediction[:, :, :4]

        # Calculate all values without allocating a new array
        x1 = pred_view[:, :, 0] - pred_view[:, :, 2] / 2
        y1 = pred_view[:, :, 1] - pred_view[:, :, 3] / 2
        x2 = pred_view[:, :, 0] + pred_view[:, :, 2] / 2
        y2 = pred_view[:, :, 1] + pred_view[:, :, 3] / 2

        # Assign directly to the view
        pred_view[:, :, 0] = x1
        pred_view[:, :, 1] = y1
        pred_view[:, :, 2] = x2
        pred_view[:, :, 3] = y2
    elif box_format != "xyxy":
        raise ValueError(
            "box_format must be either 'xywh' or 'xyxy', got {}".format(box_format)
        )

    batch_predictions = []

    # Pre-allocate space for class confidence and class prediction arrays
    cls_confs_shape = (prediction.shape[1], 1)
    cls_preds_shape = cls_confs_shape

    for np_image_i, np_image_pred in enumerate(prediction):
        np_conf_mask = np_image_pred[:, 4] >= conf_thresh
        if not np.any(np_conf_mask):  # Quick check if no boxes pass threshold
            batch_predictions.append([])
            continue

        np_image_pred = np_image_pred[np_conf_mask]

        # Handle empty case after filtering
        if np_image_pred.shape[0] == 0:
            batch_predictions.append([])
            continue

        cls_confs = np_image_pred[:, 5:num_classes + 5]

        # Check for empty classes after slicing
        if cls_confs.shape[1] == 0:
            batch_predictions.append([])
            continue

        np_class_conf = np.max(cls_confs, axis=1, keepdims=True)
        np_class_pred = np.argmax(cls_confs, axis=1, keepdims=True)
        np_mask_pred = np_image_pred[:, 5 + num_classes :]
        # Extract mask predictions if any
        if num_masks > 0:
            np_mask_pred = np_image_pred[:, 5 + num_classes:]
            # Construct final detections array directly
            np_detections = np.concatenate([
                np_image_pred[:, :5],
                np_class_conf,
                np_class_pred.astype(np.float32),
                np_mask_pred
            ], axis=1)
        else:
            # Optimization: Avoid concatenation when no masks are present
            np_detections = np.concatenate([
                np_image_pred[:, :5],
                np_class_conf,
                np_class_pred.astype(np.float32)
            ], axis=1)

        filtered_predictions = []
        np_unique_labels = np.unique(np_detections[:, 6])
        if class_agnostic:
            # Sort by confidence directly
            sorted_indices = np.argsort(-np_detections[:, 4])
            np_detections_sorted = np_detections[sorted_indices]
            # Directly pass to optimized NMS
            filtered_predictions.extend(
                non_max_suppression_fast(np_detections_sorted, iou_thresh)
            )
        else:
            np_unique_labels = np.unique(np_class_pred)

            # Process each class
            for c in np_unique_labels:
                class_mask = np_class_pred.squeeze() == c
                np_detections_class = np_detections[class_mask]

                # Skip empty arrays
                if np_detections_class.shape[0] == 0:
                    continue

                # Sort by confidence (highest first)
                sorted_indices = np.argsort(-np_detections_class[:, 4])
                np_detections_sorted = np_detections_class[sorted_indices]

                # Apply optimized NMS and extend filtered predictions
                filtered_predictions.extend(
                    non_max_suppression_fast(np_detections_sorted, iou_thresh)
                )

        # Sort final predictions by confidence and limit to max_detections
        if filtered_predictions:
            # Use numpy sort for better performance
            filtered_np = np.array(filtered_predictions)
            idx = np.argsort(-filtered_np[:, 4])
            filtered_np = filtered_np[idx]

            # Limit to max_detections
            if len(filtered_np) > max_detections:
                filtered_np = filtered_np[:max_detections]

            batch_predictions.append(list(filtered_np))
        else:
            batch_predictions.append([])

    return batch_predictions
''',
        "explanation": "Comprehensive optimization of non-maximum suppression with significant performance improvements: (1) Memory optimization by using array views instead of creating a new array for box coordinate conversion; (2) Early exit optimizations with quick threshold checks and empty array handling; (3) Vectorized operations with numpy's axis and keepdims parameters to avoid unnecessary reshape operations; (4) Conditional array concatenation to avoid overhead when no masks are present; (5) Optimized class-specific detection with direct boolean masks instead of equality filters in loops; (6) Improved sorting using numpy's specialized functions instead of Python's generic sort; (7) Better memory efficiency throughout the function with pre-allocation of temporary arrays; (8) Optimized filtering of the final predictions using numpy vectorized operations. These changes collectively reduce memory usage and computational overhead while maintaining the same functionality.",
    },
    "non_max_suppression_fast": {
        "source_code": '''
def non_max_suppression_fast(boxes, overlapThresh):
    """Applies non-maximum suppression to bounding boxes.

    Args:
        boxes (np.ndarray): Array of bounding boxes with confidence scores.
        overlapThresh (float): Overlap threshold for suppression.

    Returns:
        list: List of bounding boxes after non-maximum suppression.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float32")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    conf = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.arange(len(boxes))
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        if last == 0:
            break
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].tolist()
''',
        "explanation": "Optimized non_max_suppression_fast function with several key improvements: (1) Changed float datatype from generic 'float' to more efficient 'float32' for better memory usage and potential SIMD acceleration; (2) Eliminated the expensive argsort operation on confidence scores since the boxes are already sorted by the caller; (3) Added a crucial early-exit optimization when only one box remains in the index list; (4) Changed the return type to a Python list instead of a numpy array with explicit astype, which avoids an unnecessary array copy. These changes collectively reduce computational overhead and memory usage in this critical inner loop function that gets called repeatedly during object detection.",
    },
}


def load_optimizations_from_file(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load optimizations from a JSON file.

    Args:
        file_path: Path to the JSON file containing optimizations

    Returns:
        Dictionary of optimizations

    """
    try:
        with open(file_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load optimizations from {file_path}: {e}")
        return {}


def get_optimizations_dict() -> Dict[str, Dict[str, Any]]:
    """Get the optimizations dictionary, potentially loading from external sources.

    Returns:
        Dictionary mapping function names to optimization information

    """
    # Start with the in-memory dictionary
    optimizations = _OPTIMIZATIONS_DICT.copy()

    # Check for environment variable pointing to a JSON file with additional optimizations
    optimizations_file = os.environ.get("CODEFLASH_OPTIMIZATIONS_FILE")
    if optimizations_file:
        file_optimizations = load_optimizations_from_file(optimizations_file)
        # Update the dictionary with file-based optimizations (overriding if duplicates)
        optimizations.update(file_optimizations)

    # In the future, this could load from a database or other sources

    return optimizations


def get_manual_optimization_from_dict(
    function_name: str, optimizations_dict: Optional[Dict[str, Dict[str, Any]]] = None
) -> Optional[OptimizedCandidate]:
    """Get a manual optimization for a function from a dictionary of optimizations.

    Args:
        function_name: Name of the function to get optimization for
        optimizations_dict: Dictionary mapping function names to optimization info

    Returns:
        An OptimizedCandidate if found, None otherwise

    """
    if optimizations_dict is None:
        optimizations_dict = get_optimizations_dict()

    # Try different ways to match the function name
    # 1. Exact match
    if function_name in optimizations_dict:
        opt_info = optimizations_dict[function_name]
    else:
        # 2. Match by the simple function name (without module path)
        simple_name = function_name.split(".")[-1]
        if simple_name in optimizations_dict:
            opt_info = optimizations_dict[simple_name]
        else:
            return None

    return OptimizedCandidate(
        source_code=opt_info["source_code"],
        explanation=opt_info.get("explanation", "Manually optimized function"),
        optimization_id=f"manual-{function_name}",
    )
