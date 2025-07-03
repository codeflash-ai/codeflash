from __future__ import annotations

import base64
import json
import zlib
from copy import deepcopy
from typing import Any, Iterable
from utils import Point

from coordinates import PixelSpace
from elements import (
    TYPE_TO_TEXT_ELEMENT_MAP,
    CheckBox,
    Element,
    ElementMetadata,
)

# ================================================================================================
# SERIALIZATION/DESERIALIZATION (SERDE) RELATED FUNCTIONS
# ================================================================================================
# These serde functions will likely relocate to `unstructured.documents.elements` since they are
# so closely related to elements and this staging "brick" is deprecated.
# ================================================================================================

# == DESERIALIZERS ===============================


def elements_from_base64_gzipped_json(b64_encoded_elements: str) -> list[Element]:
    """Restore Base64-encoded gzipped JSON elements to element objects.

    This is used to when deserializing `ElementMetadata.orig_elements` from its compressed form in
    JSON and dict forms and perhaps for other purposes.
    """
    # -- Base64 str -> gzip-encoded (JSON) bytes --
    decoded_b64_bytes = base64.b64decode(b64_encoded_elements)
    # -- undo gzip compression --
    elements_json_bytes = zlib.decompress(decoded_b64_bytes)
    # -- JSON (bytes) to JSON (str) --
    elements_json_str = elements_json_bytes.decode("utf-8")
    # -- JSON (str) -> dicts --
    element_dicts = json.loads(elements_json_str)
    # -- dicts -> elements --
    return elements_from_dicts(element_dicts)


def elements_from_dicts(element_dicts: Iterable[dict[str, Any]]) -> list[Element]:
    """Convert a list of element-dicts to a list of elements."""
    elements: list[Element] = []

    for item in element_dicts:
        element_id: str = item.get("element_id", None)
        metadata = (
            ElementMetadata()
            if item.get("metadata") is None
            else ElementMetadata.from_dict(item["metadata"])
        )

        if item.get("type") in TYPE_TO_TEXT_ELEMENT_MAP:
            ElementCls = TYPE_TO_TEXT_ELEMENT_MAP[item["type"]]
            elements.append(ElementCls(text=item["text"], element_id=element_id, metadata=metadata))
        elif item.get("type") == "CheckBox":
            elements.append(
                CheckBox(checked=item["checked"], element_id=element_id, metadata=metadata)
            )

    return elements

def elements_to_base64_gzipped_json(elements: Iterable[Element]) -> str:
    """Convert `elements` to Base64-encoded gzipped JSON.

    This is used to when serializing `ElementMetadata.orig_elements` to make it as compact as
    possible when transported as JSON, for example in an HTTP response. This compressed form is also
    present when elements are in dict form ("element_dicts"). This function is not coupled to that
    purpose however and could have other uses.
    """
    # -- adjust floating-point precision of coordinates down for a more compact str value --
    precision_adjusted_elements = _fix_metadata_field_precision(elements)
    # -- serialize elements as dicts --
    element_dicts = elements_to_dicts(precision_adjusted_elements)
    # -- serialize the dicts to JSON (bytes) --
    json_bytes = json.dumps(element_dicts, sort_keys=True).encode("utf-8")
    # -- compress the JSON bytes with gzip compression --
    deflated_bytes = zlib.compress(json_bytes)
    # -- base64-encode those bytes so they can be serialized as a JSON string value --
    b64_deflated_bytes = base64.b64encode(deflated_bytes)
    # -- convert to a string suitable for serializing in JSON --
    return b64_deflated_bytes.decode("utf-8")


def elements_to_dicts(elements: Iterable[Element]) -> list[dict[str, Any]]:
    """Convert document elements to element-dicts."""
    return [e.to_dict() for e in elements]


def _fix_metadata_field_precision(elements: Iterable[Element]) -> list[Element]:
    out_elements: list[Element] = []
    for element in elements:
        el = deepcopy(element)
        if el.metadata.coordinates:
            precision = 1 if isinstance(el.metadata.coordinates.system, PixelSpace) else 2
            points = el.metadata.coordinates.points
            assert points is not None
            rounded_points: list[Point] = []
            for point in points:
                x, y = point
                rounded_point = (round(x, precision), round(y, precision))
                rounded_points.append(rounded_point)
            el.metadata.coordinates.points = tuple(rounded_points)

        if el.metadata.detection_class_prob:
            el.metadata.detection_class_prob = round(el.metadata.detection_class_prob, 5)

        out_elements.append(el)

    return out_elements