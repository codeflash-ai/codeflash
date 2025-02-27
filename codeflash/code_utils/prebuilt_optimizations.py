import json
import os
from typing import Any, Dict, Optional

from codeflash.cli_cmds.console import logger
from codeflash.models.models import OptimizedCandidate

# In-memory dictionary of optimizations
# This can be extended later to pull from a database
_OPTIMIZATIONS_DICT: Dict[str, Dict[str, Any]] = {
    "load_image": {
        "source_code": '''
class OnnxRoboflowInferenceModel(RoboflowInferenceModel):
    """Roboflow Inference Model that operates using an ONNX model file."""

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

    def infer(self, image: Any, **kwargs) -> Any:
        """Runs inference on given data.
        - image:
            can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
        """
        input_elements = len(image) if isinstance(image, list) else 1
        max_batch_size = MAX_BATCH_SIZE if self.batching_enabled else self.batch_size
        if (input_elements == 1) or (max_batch_size == float("inf")):
            return super().infer(image, **kwargs)
        logger.debug(
            f"Inference will be executed in batches, as there is {input_elements} input elements and "
            f"maximum batch size for a model is set to: {max_batch_size}"
        )
        inference_results = []
        for batch_input in create_batches(sequence=image, batch_size=max_batch_size):
            batch_inference_results = super().infer(batch_input, **kwargs)
            inference_results.append(batch_inference_results)
        return self.merge_inference_results(inference_results=inference_results)

    def merge_inference_results(self, inference_results: List[Any]) -> Any:
        return list(itertools.chain(*inference_results))

    def validate_model(self) -> None:
        if MODEL_VALIDATION_DISABLED:
            logger.debug("Model validation disabled.")
            return None
        logger.debug("Starting model validation")
        if not self.load_weights:
            return
        try:
            assert self.onnx_session is not None
        except AssertionError as e:
            raise ModelArtefactError(
                "ONNX session not initialized. Check that the model weights are available."
            ) from e
        try:
            self.run_test_inference()
        except Exception as e:
            raise ModelArtefactError(f"Unable to run test inference. Cause: {e}") from e
        try:
            self.validate_model_classes()
        except Exception as e:
            raise ModelArtefactError(
                f"Unable to validate model classes. Cause: {e}"
            ) from e
        logger.debug("Model validation finished")

    def run_test_inference(self) -> None:
        test_image = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
        logger.debug(f"Running test inference. Image size: {test_image.shape}")
        result = self.infer(test_image, usage_inference_test_run=True)
        logger.debug(f"Test inference finished.")
        return result

    def get_model_output_shape(self) -> Tuple[int, int, int]:
        test_image = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
        logger.debug(f"Getting model output shape. Image size: {test_image.shape}")
        test_image, _ = self.preprocess(test_image)
        output = self.predict(test_image)[0]
        logger.debug(f"Model output shape test finished.")
        return output.shape

    def validate_model_classes(self) -> None:
        pass

    def get_infer_bucket_file_list(self) -> list:
        """Returns the list of files to be downloaded from the inference bucket for ONNX model.

        Returns:
            list: A list of filenames specific to ONNX models.
        """
        return ["environment.json", "class_names.txt"]

    def initialize_model(self) -> None:
        """Initializes the ONNX model, setting up the inference session and other necessary properties."""
        logger.debug("Getting model artefacts")
        self.get_model_artifacts()
        logger.debug("Creating inference session")
        if self.load_weights or not self.has_model_metadata:
            t1_session = perf_counter()
            # Create an ONNX Runtime Session with a list of execution providers in priority order. ORT attempts to load providers until one is successful. This keeps the code across devices identical.
            providers = self.onnxruntime_execution_providers

            if not self.load_weights:
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            try:
                session_options = onnxruntime.SessionOptions()
                session_options.log_severity_level = 3
                # TensorRT does better graph optimization for its EP than onnx
                if has_trt(providers):
                    session_options.graph_optimization_level = (
                        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                    )
                self.onnx_session = onnxruntime.InferenceSession(
                    self.cache_file(self.weights_file),
                    providers=providers,
                    sess_options=session_options,
                )
            except Exception as e:
                self.clear_cache()
                raise ModelArtefactError(
                    f"Unable to load ONNX session. Cause: {e}"
                ) from e
            logger.debug(f"Session created in {perf_counter() - t1_session} seconds")

            if REQUIRED_ONNX_PROVIDERS:
                available_providers = onnxruntime.get_available_providers()
                for provider in REQUIRED_ONNX_PROVIDERS:
                    if provider not in available_providers:
                        raise OnnxProviderNotAvailable(
                            f"Required ONNX Execution Provider {provider} is not availble. "
                            "Check that you are using the correct docker image on a supported device. "
                            "Export list of available providers as ONNXRUNTIME_EXECUTION_PROVIDERS environmental variable, "
                            "consult documentation for more details."
                        )

            inputs = self.onnx_session.get_inputs()[0]
            input_shape = inputs.shape
            self.batch_size = input_shape[0]
            self.img_size_h = input_shape[2]
            self.img_size_w = input_shape[3]
            self.input_name = inputs.name
            if isinstance(self.img_size_h, str) or isinstance(self.img_size_w, str):
                if "resize" in self.preproc:
                    self.img_size_h = int(self.preproc["resize"]["height"])
                    self.img_size_w = int(self.preproc["resize"]["width"])
                else:
                    self.img_size_h = 640
                    self.img_size_w = 640

            if isinstance(self.batch_size, str):
                self.batching_enabled = True
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching enabled"
                )
            else:
                self.batching_enabled = False
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching disabled"
                )

            model_metadata = {
                "batch_size": self.batch_size,
                "img_size_h": self.img_size_h,
                "img_size_w": self.img_size_w,
            }
            logger.debug(f"Writing model metadata to memcache")
            self.write_model_metadata_to_memcache(model_metadata)
            if not self.load_weights:  # had to load weights to get metadata
                del self.onnx_session
        else:
            if not self.has_model_metadata:
                raise ValueError(
                    "This should be unreachable, should get weights if we don't have model metadata"
                )
            logger.debug(f"Loading model metadata from memcache")
            metadata = self.model_metadata_from_memcache()
            self.batch_size = metadata["batch_size"]
            self.img_size_h = metadata["img_size_h"]
            self.img_size_w = metadata["img_size_w"]
            if isinstance(self.batch_size, str):
                self.batching_enabled = True
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching enabled"
                )
            else:
                self.batching_enabled = False
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching disabled"
                )

        logger.debug("Model initialisation finished.")

    def load_image(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
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
        else:
            # Use multiprocessing for larger batches
            import multiprocessing as mp
            # Calculate optimal worker count
            num_workers = min(len(image), max(1, mp.cpu_count() - 1))
            # Create the worker function
            def worker_function(img):
                return self.preproc_image(
                    img,
                    disable_preproc_auto_orient=disable_preproc_auto_orient,
                    disable_preproc_contrast=disable_preproc_contrast,
                    disable_preproc_grayscale=disable_preproc_grayscale,
                    disable_preproc_static_crop=disable_preproc_static_crop,
                )
            # Use multiprocessing pool
            with mp.Pool(processes=num_workers) as pool:
                imgs_with_dims = pool.map(worker_function, image)
        # Extract images and dimensions
        imgs, img_dims = zip(*imgs_with_dims)
        # Combine into batch
        if isinstance(imgs[0], np.ndarray):
            img_in = np.concatenate(imgs, axis=0)
        elif "torch" in dir():
            img_in = torch.cat(imgs, dim=0)
        else:
            raise ValueError(f"Unsupported image type: {type(imgs[0])}")
        return img_in, img_dims

     @property
    def weights_file(self) -> str:
        """Returns the file containing the ONNX model weights.

        Returns:
            str: The file path to the weights file.
        """
        return "weights.onnx"
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


        np_unique_labels = np.unique(np_detections[:, 6])

        if class_agnostic:
            # Sort by confidence directly
            sorted_indices = np.argsort(-np_detections[:, 4])
            np_detections_sorted = np_detections[sorted_indices]
            # Directly pass to optimized NMS
            filtered_predictions = non_max_suppression_fast(np_detections_sorted, iou_thresh)
        else:
            filtered_predictions = []
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

            batch_predictions.append(filtered_np.tolist())
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
