from __future__ import annotations

import numpy as np
import pyarrow as pa


INSTRUCTION_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("task", pa.string()),
        ("mode", pa.string()),
        ("difficulty", pa.string()),
        ("dataset_id", pa.string()),
        ("domain", pa.string()),
        ("language", pa.string()),
        ("split", pa.string()),
        (
            "messages",
            pa.list_(
                pa.struct(
                    [
                        ("role", pa.string()),
                        ("content", pa.string()),
                    ]
                )
            ),
        ),
        ("prompt", pa.string()),
        ("response", pa.string()),
        ("rendered_text", pa.string()),
        ("source_keys", pa.list_(pa.string())),
        ("evidence", pa.list_(pa.string())),
        ("answer_json", pa.string()),
        ("source", pa.string()),
        ("source_urls", pa.list_(pa.string())),
        ("license", pa.string()),
        ("version", pa.string()),
    ]
)


PROJECTED_SFT_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("task", pa.string()),
        ("mode", pa.string()),
        ("difficulty", pa.string()),
        ("domain", pa.string()),
        ("language", pa.string()),
        ("split", pa.string()),
        (
            "messages",
            pa.list_(
                pa.struct(
                    [
                        ("role", pa.string()),
                        ("content", pa.string()),
                    ]
                )
            ),
        ),
        ("prompt", pa.string()),
        ("response", pa.string()),
        ("reasoning_envelope", pa.bool_()),
        ("reasoning_trace", pa.string()),
        ("final_response", pa.string()),
        ("reasoning_card_hand", pa.string()),
        ("structure_signature", pa.string()),
        ("response_card_hand", pa.string()),
        ("source_representation", pa.string()),
        ("source", pa.string()),
        ("license", pa.string()),
        ("version", pa.string()),
    ]
)


TOKEN_DTYPE = np.dtype("<u4")


LABEL_DTYPE = np.dtype("<i4")


IGNORE_INDEX = -100
