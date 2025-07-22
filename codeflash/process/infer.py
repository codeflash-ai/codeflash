import numpy as np


def sigmoid_stable(x):
    exp_neg_x = np.exp(-x)
    exp_x = np.exp(x)
    # Use precomputed exponentials to avoid redundant calculation
    return np.where(x >= 0, 1 / (1 + exp_neg_x), exp_x / (1 + exp_x))


def postprocess(logits: np.array, max_detections: int = 8):
    batch_size, num_queries, num_classes = logits.shape
    logits_sigmoid = sigmoid_stable(logits)
    processed_predictions = []
    for batch_idx in range(batch_size):
        logits_flat = logits_sigmoid[batch_idx].reshape(-1)

        sorted_indices = np.argsort(-logits_flat)[:max_detections]
        processed_predictions.append(sorted_indices)
    return processed_predictions


if __name__ == "__main__":
    predictions = np.random.normal(size=(8, 1000, 10))
    print(predictions.shape)
    result = postprocess(predictions, max_detections=8)
    print(len(result), result[0])
