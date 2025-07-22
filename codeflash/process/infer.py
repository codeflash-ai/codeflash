import numpy as np


def sigmoid_stable(x):
    # Avoid repeated computation of exp(x)
    ex = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + ex), ex / (1 + ex))


def postprocess(logits: np.array, max_detections: int = 8):
    batch_size, num_queries, num_classes = logits.shape
    logits_sigmoid = sigmoid_stable(logits)
    # Preallocate output as an array for efficiency
    processed_predictions = [None] * batch_size
    for batch_idx in range(batch_size):
        logits_flat = logits_sigmoid[batch_idx].ravel()
        if logits_flat.size <= max_detections:
            # If there are fewer elements than max_detections, just argsort all
            sorted_indices = np.argsort(-logits_flat)
        else:
            # Partial sort for top max_detections
            partition_indices = np.argpartition(-logits_flat, max_detections - 1)[:max_detections]
            top_scores = logits_flat[partition_indices]
            # Now sort these to get actual order
            sorted_order = np.argsort(-top_scores)
            sorted_indices = partition_indices[sorted_order]
        processed_predictions[batch_idx] = sorted_indices
    return processed_predictions


if __name__ == "__main__":
    predictions = np.random.normal(size=(8, 1000, 10))
    print(predictions.shape)
    result = postprocess(predictions, max_detections=8)
    print(len(result), result[0])
