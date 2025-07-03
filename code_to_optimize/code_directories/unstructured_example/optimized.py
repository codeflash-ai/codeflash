from __future__ import annotations
import os
import pathlib
from typing import Any, Iterable, Optional

from elements import (TYPE_TO_TEXT_ELEMENT_MAP, CheckBox,
                                       Element)
from elements import ElementMetadata as _ElementMetadata


# Helper to resolve 'pathlib.Path' on filename efficiently
def _extract_file_directory_and_name(filename: Optional[str | pathlib.Path], file_directory: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if isinstance(filename, pathlib.Path):
        filename = str(filename)
    directory_path, file_name = os.path.split(filename or "")
    return (file_directory or directory_path or None, file_name or None)

class ElementMetadata:
    def __init__(
            self,
        attached_to_filename: Optional[str] = None,
        bcc_recipient: Optional[list[str]] = None,
        category_depth: Optional[int] = None,
        cc_recipient: Optional[list[str]] = None,
        coordinates: Optional[Any] = None,
        data_source: Optional[Any] = None,
        detection_class_prob: Optional[float] = None,
        emphasized_text_contents: Optional[list[str]] = None,
        emphasized_text_tags: Optional[list[str]] = None,
        file_directory: Optional[str] = None,
        filename: Optional[str | pathlib.Path] = None,
        filetype: Optional[str] = None,
        header_footer_type: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_mime_type: Optional[str] = None,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        is_continuation: Optional[bool] = None,
        languages: Optional[list[str]] = None,
        last_modified: Optional[str] = None,
        link_start_indexes: Optional[list[int]] = None,
        link_texts: Optional[list[str]] = None,
        link_urls: Optional[list[str]] = None,
        links: Optional[list[Any]] = None,
        email_message_id: Optional[str] = None,
        orig_elements: Optional[list[Element]] = None,
        page_name: Optional[str] = None,
        page_number: Optional[int] = None,
        parent_id: Optional[str] = None,
        sent_from: Optional[list[str]] = None,
        sent_to: Optional[list[str]] = None,
        signature: Optional[str] = None,
        subject: Optional[str] = None,
        table_as_cells: Optional[dict[str, str | int]] = None,
        text_as_html: Optional[str] = None,
        url: Optional[str] = None,
        key_value_pairs: Optional[list[Any]] = None,
    ) -> None:
        self.attached_to_filename = attached_to_filename
        self.bcc_recipient = bcc_recipient
        self.category_depth = category_depth
        self.cc_recipient = cc_recipient
        self.coordinates = coordinates
        self.data_source = data_source
        self.detection_class_prob = detection_class_prob
        self.emphasized_text_contents = emphasized_text_contents
        self.emphasized_text_tags = emphasized_text_tags

        # -- accommodate pathlib.Path for filename --
        self.file_directory, self.filename = _extract_file_directory_and_name(filename, file_directory)
        self.filetype = filetype
        self.header_footer_type = header_footer_type
        self.image_base64 = image_base64
        self.image_mime_type = image_mime_type
        self.image_url = image_url
        self.image_path = image_path
        self.is_continuation = is_continuation
        self.languages = languages
        self.last_modified = last_modified
        self.link_texts = link_texts
        self.link_urls = link_urls
        self.link_start_indexes = link_start_indexes
        self.links = links
        self.email_message_id = email_message_id
        self.orig_elements = orig_elements
        self.page_name = page_name
        self.page_number = page_number
        self.parent_id = parent_id
        self.sent_from = sent_from
        self.sent_to = sent_to
        self.signature = signature
        self.subject = subject
        self.text_as_html = text_as_html
        self.table_as_cells = table_as_cells
        self.url = url
        self.key_value_pairs = key_value_pairs

    @classmethod
    def from_dict(cls, meta_dict: dict[str, Any]) -> 'ElementMetadata':
        """Construct from a metadata-dict.

        This would generally be a dict formed using the `.to_dict()` method and stored as JSON
        before "rehydrating" it using this method.
        """
        from base import elements_from_base64_gzipped_json

        # Rather than copy.deepcopy, build fast new fields dict
        key_value = meta_dict.get

        # Local import avoids import cycles (as originally intended)
        coords_val = key_value("coordinates")
        coordinates = CoordinatesMetadata.from_dict(coords_val) if coords_val is not None else None

        data_source_val = key_value("data_source")
        data_source = DataSourceMetadata.from_dict(data_source_val) if data_source_val is not None else None

        orig_elements_val = key_value("orig_elements")
        orig_elements = (
                elements_from_base64_gzipped_json(orig_elements_val) if orig_elements_val is not None else None
        )

        key_value_pairs_val = key_value("key_value_pairs")
        key_value_pairs = (
                _kvform_rehydrate_internal_elements(key_value_pairs_val) if key_value_pairs_val is not None else None
        )

        # Build argument dict for __init__ using known args and self assignment for remaining
        # Fast field assignment - all remaining fields
        args = {
                k: v for k, v in meta_dict.items()
            if k not in ("coordinates", "data_source", "orig_elements", "key_value_pairs")
        }
        args["coordinates"] = coordinates
        args["data_source"] = data_source
        args["orig_elements"] = orig_elements
        args["key_value_pairs"] = key_value_pairs

        return cls(**args)

def elements_from_dicts(element_dicts: Iterable[dict[str, Any]]) -> list[Element]:
    """Convert a list of element-dicts to a list of elements."""
    # Localize references for speed
    ETM_MAP = TYPE_TO_TEXT_ELEMENT_MAP
    result_append = []
    result_append_method = result_append.append  # avoid attribute lookup inside loop
    CheckBoxClass = CheckBox

    for item in element_dicts:
        itype = item.get("type")
        element_id = item.get("element_id")
        meta = item.get("metadata")
        metadata = ElementMetadata() if meta is None else ElementMetadata.from_dict(meta)
        if itype in ETM_MAP:
            ElementCls = ETM_MAP[itype]
            result_append_method(ElementCls(
                    text=item["text"], element_id=element_id, metadata=metadata
            ))
        elif itype == "CheckBox":
            result_append_method(
                    CheckBoxClass(
                        checked=item["checked"], element_id=element_id, metadata=metadata
                )
            )
    return result_append

