def _equalize_cv(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    i = 0
    for val in histogram:
        if val > 0:
            break
        i += 1
    i = min(i, 255)

    total = np.sum(histogram)
    if histogram[i] == total:
        return np.full_like(img, i)

    scale = 255.0 / (total - histogram[i])
    _sum = 0

    lut = np.zeros(256, dtype=np.uint8)

    for idx in range(i + 1, len(histogram)):
        _sum += histogram[idx]
        lut[idx] = clip(round(_sum * scale), np.uint8)
    return sz_lut(img, lut, inplace=True)
