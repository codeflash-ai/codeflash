def _equalize_cv(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()

    # Find the first non-zero index with a numpy operation
    i = np.flatnonzero(histogram)[0] if np.any(histogram) else 255

    total = np.sum(histogram)
    if histogram[i] == total:
        return np.full_like(img, i)

    scale = 255.0 / (total - histogram[i])

    # Optimize cumulative sum and scale to generate LUT
    cumsum_histogram = np.cumsum(histogram)
    lut = np.clip(((cumsum_histogram - cumsum_histogram[i]) * scale)
                  .round(), 0, 255).astype(np.uint8)
    return sz_lut(img, lut, inplace=True)
