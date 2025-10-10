"""Data type codec for CacheManager serialization.

This module provides a unified codec for encoding/decoding pandas DataFrame values
to/from JSON Schema format (v2), with explicit type information and no type inference.

Key features:
- JSON Schema v2 format: {"v": 2, "val": ..., "type": ..., "meta": null, "special": null}
- Explicit type handling (no type inference/guessing)
- Minimal fallbacks (< 5), all documented
- Special value support: np.nan, pd.NA, None, np.inf, -np.inf
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

import json

logger = logging.getLogger(__name__)


@dataclass
class EncodedValue:
    """Encoded value with type information.

    Attributes:
        value: The actual value (can be None for special values)
        dtype: numpy/pandas dtype string (e.g., 'int64', 'float64', 'datetime64[ns]')
        metadata: Additional metadata (e.g., categories for categorical dtype)
        special: Special value type ('nan', 'pd_na', 'none', 'inf', '-inf', or None)
    """
    value: Any
    dtype: str
    metadata: Optional[dict] = None
    special: Optional[str] = None


class DataTypeCodec:
    """Unified data type codec for serialization/deserialization.

    This codec converts pandas/numpy values to JSON Schema v2 format, which is
    self-contained and requires no external type inference.

    Version: 2.0
    Format: {"v": 2, "val": <value>, "type": <dtype>, "meta": <metadata>, "special": <special>}

    Example:
        >>> codec = DataTypeCodec()
        >>> encoded = codec.encode(123, dtype='int64')
        >>> print(encoded)
        '{"v":2,"val":123,"type":"int64","meta":null,"special":null}'
        >>> decoded = codec.decode(encoded)
        >>> print(decoded, type(decoded))
        123 <class 'numpy.int64'>
    """

    VERSION = 2

    # Special value markers
    SPECIAL_NAN = "nan"
    SPECIAL_PD_NA = "pd_na"
    SPECIAL_NONE = "none"
    SPECIAL_INF = "inf"
    SPECIAL_NEG_INF = "-inf"

    def __init__(self):
        """Initialize codec."""
        pass

    def _dumps_json(self, obj: Any) -> str:
        """Serialize object to JSON string with orjson fallback.

        Fallback #1: orjson → json (performance optimization)
        Trigger: orjson not available or raises JSONEncodeError

        Args:
            obj: Object to serialize

        Returns:
            JSON string
        """
        if HAS_ORJSON:
            try:
                return orjson.dumps(obj).decode("utf-8")
            except (TypeError, orjson.JSONEncodeError) as e:
                logger.warning(f"orjson encoding failed: {e}, falling back to json")

        return json.dumps(obj, ensure_ascii=False)

    def _loads_json(self, json_str: str) -> Any:
        """Deserialize JSON string with orjson fallback.

        Fallback #2: orjson → json (performance optimization)
        Trigger: orjson not available or raises JSONDecodeError

        Args:
            json_str: JSON string to parse

        Returns:
            Parsed object
        """
        if HAS_ORJSON:
            try:
                return orjson.loads(json_str)
            except (orjson.JSONDecodeError, TypeError) as e:
                logger.warning(f"orjson decoding failed: {e}, falling back to json")

        return json.loads(json_str)

    def _detect_special_value(self, value: Any) -> Optional[str]:
        """Detect if value is a special value and return its marker.

        Args:
            value: Value to check

        Returns:
            Special value marker string, or None if not special

        Note:
            This is NOT type inference - we explicitly check for known special values.
        """
        # pd.NA check (must be before pd.isna since pd.NA is considered NA)
        if pd.isna(value):
            # Distinguish between np.nan and pd.NA
            if hasattr(pd, 'NA') and value is pd.NA:
                return self.SPECIAL_PD_NA
            # Check for np.nan
            if isinstance(value, float) and np.isnan(value):
                return self.SPECIAL_NAN
            # Check for np.inf
            if isinstance(value, float) and np.isinf(value):
                return self.SPECIAL_INF if value > 0 else self.SPECIAL_NEG_INF

        # None check
        if value is None:
            return self.SPECIAL_NONE

        # np.inf check (additional, in case pd.isna doesn't catch it)
        if isinstance(value, (float, np.floating)):
            if np.isinf(value):
                return self.SPECIAL_INF if value > 0 else self.SPECIAL_NEG_INF
            if np.isnan(value):
                return self.SPECIAL_NAN

        return None

    def _infer_dtype(self, value: Any) -> str:
        """Infer dtype string from value.

        This is used when dtype is not explicitly provided.
        We map to specific numpy dtype strings.

        Args:
            value: Value to infer dtype from

        Returns:
            dtype string (e.g., 'int64', 'float64', 'object')
        """
        # Handle special values
        if self._detect_special_value(value) is not None:
            if isinstance(value, float) or isinstance(value, np.floating):
                return "float64"
            return "object"

        # Numpy types
        if isinstance(value, np.integer):
            return str(value.dtype)  # 'int8', 'int16', 'int32', 'int64', etc.
        if isinstance(value, np.floating):
            return str(value.dtype)  # 'float16', 'float32', 'float64'
        if isinstance(value, np.bool_):
            return "bool"

        # Python native types -> numpy dtypes
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int64"  # Default to int64 for Python int
        if isinstance(value, float):
            return "float64"
        if isinstance(value, str):
            return "object"

        # pandas types
        if isinstance(value, pd.Timestamp):
            return "datetime64[ns]"

        # Fallback to object
        return "object"

    def _convert_to_json_compatible(self, value: Any) -> Any:
        """Convert numpy/pandas types to JSON-compatible Python types.

        Args:
            value: Value to convert

        Returns:
            JSON-compatible value (Python int, float, bool, str, etc.)
        """
        if value is None or value is pd.NA:
            return None

        # Numpy types → Python types
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)

        # pandas Timestamp → ISO string
        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        # Already JSON-compatible
        if isinstance(value, (int, float, bool, str)):
            return value

        # Fallback to string
        return str(value)

    def encode(self, value: Any, dtype: Optional[str] = None) -> str:
        """Encode a single value to JSON Schema v2 format.

        Args:
            value: Value to encode
            dtype: Explicit dtype string (if known). If None, will be inferred.

        Returns:
            JSON string in v2 format

        Example:
            >>> codec.encode(123, dtype='int64')
            '{"v":2,"val":123,"type":"int64","meta":null,"special":null}'
            >>> codec.encode(np.nan, dtype='float64')
            '{"v":2,"val":null,"type":"float64","meta":null,"special":"nan"}'
        """
        # Detect special values
        special_marker = self._detect_special_value(value)

        # Infer dtype if not provided
        if dtype is None:
            dtype = self._infer_dtype(value)

        # Convert value to JSON-compatible format
        json_value = None if special_marker else self._convert_to_json_compatible(value)

        # Build schema
        schema = {
            "v": self.VERSION,
            "val": json_value,
            "type": dtype,
            "meta": None,
            "special": special_marker
        }

        # Serialize to JSON (with orjson fallback)
        return self._dumps_json(schema)

    def decode(self, encoded: str) -> Any:
        """Decode a JSON Schema v2 string to original value.

        Args:
            encoded: JSON string in v2 format

        Returns:
            Decoded value with correct type

        Raises:
            ValueError: If encoded string is invalid or unsupported format

        Example:
            >>> codec.decode('{"v":2,"val":123,"type":"int64",...}')
            123  # as np.int64
        """
        # Parse JSON (with orjson fallback)
        schema = self._loads_json(encoded)

        # Validate schema version
        if not isinstance(schema, dict) or schema.get("v") != self.VERSION:
            raise ValueError(f"Invalid schema version or format: {schema}")

        # Extract fields
        value = schema.get("val")
        dtype_str = schema.get("type")
        special = schema.get("special")
        # metadata = schema.get("meta")  # Reserved for future use (e.g., categorical)

        # Handle special values first
        if special:
            if special == self.SPECIAL_NAN:
                return np.nan
            elif special == self.SPECIAL_PD_NA:
                return pd.NA
            elif special == self.SPECIAL_NONE:
                return None
            elif special == self.SPECIAL_INF:
                return np.inf
            elif special == self.SPECIAL_NEG_INF:
                return -np.inf
            else:
                raise ValueError(f"Unknown special value marker: {special}")

        # Handle normal values with explicit type conversion
        if value is None:
            return None

        # Convert based on dtype
        if dtype_str.startswith("int"):
            # int8, int16, int32, int64, uint8, uint16, uint32, uint64
            return np.dtype(dtype_str).type(value)
        elif dtype_str.startswith("uint"):
            return np.dtype(dtype_str).type(value)
        elif dtype_str.startswith("float"):
            # float16, float32, float64
            return np.dtype(dtype_str).type(value)
        elif dtype_str == "bool":
            return np.bool_(value)
        elif dtype_str.startswith("datetime64"):
            # Fallback #3: pd.Timestamp → datetime conversion
            # Trigger: Timestamp value out of pandas range
            try:
                return pd.Timestamp(value)
            except pd.errors.OutOfBoundsDatetime as e:
                # Use standard datetime for out-of-range values
                from datetime import datetime
                logger.warning(f"Timestamp out of pandas range: {e}, using datetime")
                return datetime.fromisoformat(value)
        elif dtype_str == "object":
            # Keep as-is (string or other object)
            return value
        else:
            # Unsupported dtype - this should not happen if encoding is correct
            raise ValueError(f"Unsupported dtype for decoding: {dtype_str}")

    def encode_row(self, row_dict: dict, dtypes: Optional[dict] = None) -> str:
        """Encode an entire row (dict of column -> value) to JSON.

        Args:
            row_dict: Dictionary mapping column names to values
            dtypes: Optional dictionary mapping column names to dtype strings

        Returns:
            JSON string encoding the entire row in v2 format

        Example:
            >>> codec.encode_row({"A": 1, "B": 2.5, "C": np.nan})
            '{"v":2,"d":{"A":{"val":1,"type":"int64",...},...}}'
        """
        dtypes = dtypes or {}

        encoded_data = {}
        for col, value in row_dict.items():
            dtype = dtypes.get(col)
            # Parse the individual encoded value
            individual_encoded = self.encode(value, dtype=dtype)
            # Remove the outer JSON layer (we'll nest it under "d")
            individual_schema = json.loads(individual_encoded)
            # Store without the "v" key (version is at row level)
            encoded_data[col] = {
                "val": individual_schema["val"],
                "type": individual_schema["type"],
                "meta": individual_schema.get("meta"),
                "special": individual_schema.get("special")
            }

        # Build row schema
        row_schema = {
            "v": self.VERSION,
            "d": encoded_data
        }

        # Serialize (with orjson fallback)
        return self._dumps_json(row_schema)

    def decode_row(self, json_str: str) -> dict:
        """Decode a JSON string to a row dictionary.

        Args:
            json_str: JSON string in row v2 format

        Returns:
            Dictionary mapping column names to decoded values

        Example:
            >>> codec.decode_row('{"v":2,"d":{"A":{"val":1,"type":"int64",...}}}')
            {"A": 1, "B": 2.5, ...}
        """
        # Parse JSON (with orjson fallback)
        row_schema = self._loads_json(json_str)

        # Validate
        if not isinstance(row_schema, dict) or row_schema.get("v") != self.VERSION:
            raise ValueError(f"Invalid row schema version or format: {row_schema}")

        data = row_schema.get("d", {})

        # Decode each column
        decoded_row = {}
        for col, col_schema in data.items():
            # Reconstruct individual value schema
            value_schema = {
                "v": self.VERSION,
                "val": col_schema.get("val"),
                "type": col_schema.get("type"),
                "meta": col_schema.get("meta"),
                "special": col_schema.get("special")
            }
            # Decode
            decoded_row[col] = self.decode(json.dumps(value_schema))

        return decoded_row
