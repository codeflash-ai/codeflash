def postprocess(self, predictions: tuple[np.ndarray, ...], max_detections: int):
    bboxes, logits = predictions
    batch_size, num_queries, num_classes = logits.shape
    logits_sigmoid = self.sigmoid_stable(logits)
    for batch_idx in range(batch_size):
        logits_flat = logits_sigmoid[batch_idx].reshape(-1)
        # Use argpartition for better performance when max_detections is smaller than logits_flat
        partition_indices = np.argpartition(-logits_flat, max_detections)[:max_detections]
        sorted_indices = partition_indices[np.argsort(-logits_flat[partition_indices])]
