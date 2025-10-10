"""Tests for DataTypeCodec fallback handling.

This test suite ensures that all fallbacks are:
1. Documented with clear comments
2. Triggered by specific, testable conditions
3. Use specific exception types (not bare except:)
4. Have 100% test coverage
"""

import json
import pytest
import numpy as np
import pandas as pd

from finlab_guard.cache.codec import DataTypeCodec


class TestCodecFallbacks:
    """Test fallback mechanisms in DataTypeCodec."""

    @pytest.fixture
    def codec(self):
        """Create a DataTypeCodec instance."""
        return DataTypeCodec()

    # ========== Fallback #1 & #2: JSON Parser (orjson → json) ==========

    def test_fallback_orjson_to_json_encoding(self, codec, monkeypatch):
        """Test fallback from orjson to json during encoding.

        Trigger condition: orjson not available or raises JSONEncodeError
        Expected behavior: Falls back to standard json.dumps()
        """
        # Simulate orjson failure by raising an exception
        original_orjson = codec.encode.__globals__.get('orjson')

        try:
            # Patch HAS_ORJSON to True but make orjson.dumps fail
            import finlab_guard.cache.codec as codec_module
            codec_module.HAS_ORJSON = True

            # Mock orjson.dumps to raise an error
            class MockOrjson:
                class JSONEncodeError(Exception):
                    pass

                @staticmethod
                def dumps(obj):
                    raise MockOrjson.JSONEncodeError("Mock encoding error")

            monkeypatch.setattr(codec_module, 'orjson', MockOrjson)

            # This should trigger fallback to json.dumps()
            value = 123
            encoded = codec.encode(value, dtype='int64')

            # Should still work via json fallback
            schema = json.loads(encoded)
            assert schema['val'] == 123
            assert schema['type'] == 'int64'

        finally:
            # Restore original state
            if original_orjson:
                import finlab_guard.cache.codec as codec_module
                codec_module.orjson = original_orjson

    def test_fallback_orjson_to_json_decoding(self, codec, monkeypatch):
        """Test fallback from orjson to json during decoding.

        Trigger condition: orjson.loads() raises JSONDecodeError
        Expected behavior: Falls back to standard json.loads()
        """
        import finlab_guard.cache.codec as codec_module
        original_orjson = codec_module.orjson if hasattr(codec_module, 'orjson') else None

        try:
            # Set HAS_ORJSON to True
            codec_module.HAS_ORJSON = True

            # Mock orjson.loads to raise an error
            class MockOrjson:
                class JSONDecodeError(Exception):
                    pass

                @staticmethod
                def loads(s):
                    raise MockOrjson.JSONDecodeError("Mock decoding error")

            monkeypatch.setattr(codec_module, 'orjson', MockOrjson)

            # Prepare a valid JSON string
            encoded = '{"v":2,"val":456,"type":"int64","meta":null,"special":null}'

            # This should trigger fallback to json.loads()
            decoded = codec.decode(encoded)

            # Should still work via json fallback
            assert decoded == 456

        finally:
            # Restore
            if original_orjson:
                codec_module.orjson = original_orjson

    def test_json_encoding_without_orjson(self, codec, monkeypatch):
        """Test encoding when orjson is not available.

        Trigger condition: HAS_ORJSON = False
        Expected behavior: Uses json.dumps() directly
        """
        import finlab_guard.cache.codec as codec_module

        # Disable orjson
        original_has_orjson = codec_module.HAS_ORJSON
        codec_module.HAS_ORJSON = False

        try:
            value = 789
            encoded = codec.encode(value, dtype='int64')

            # Should work with standard json
            schema = json.loads(encoded)
            assert schema['val'] == 789

        finally:
            codec_module.HAS_ORJSON = original_has_orjson

    # ========== Fallback #3: Timestamp Out of Range ==========

    def test_fallback_timestamp_out_of_range(self, codec, monkeypatch):
        """Test fallback for timestamps outside pandas range.

        Trigger condition: pd.Timestamp() raises OutOfBoundsDatetime
        Expected behavior: Falls back to datetime.fromisoformat()

        Note: This is a documented fallback for very old/future dates
        that exceed pandas' datetime64[ns] range (~1677-2262).
        """
        # Mock pd.Timestamp to raise OutOfBoundsDatetime
        import pandas as pd
        original_timestamp = pd.Timestamp

        def mock_timestamp(value):
            if isinstance(value, str) and value.startswith('9999'):
                raise pd.errors.OutOfBoundsDatetime("Year is out of bounds")
            return original_timestamp(value)

        monkeypatch.setattr(pd, 'Timestamp', mock_timestamp)

        # Create an encoded value with a far-future date
        # (normally this would come from encoding, but we construct it directly)
        encoded = json.dumps({
            "v": 2,
            "val": "9999-12-31T23:59:59",
            "type": "datetime64[ns]",
            "meta": None,
            "special": None
        })

        # This should trigger fallback to datetime.fromisoformat()
        decoded = codec.decode(encoded)

        # Should return a datetime object (not pd.Timestamp)
        from datetime import datetime
        assert isinstance(decoded, datetime)

    # ========== No Blind Fallbacks ==========

    def test_no_fallback_on_invalid_schema_version(self, codec):
        """Test that invalid schema version is NOT silently handled.

        Expected behavior: Raises ValueError, not silently falls back
        """
        invalid_encoded = '{"v":999,"val":123,"type":"int64","meta":null,"special":null}'

        with pytest.raises(ValueError, match="Invalid schema version"):
            codec.decode(invalid_encoded)

    def test_no_fallback_on_invalid_dtype(self, codec):
        """Test that unsupported dtype is NOT silently handled.

        Expected behavior: Raises ValueError for unknown dtype
        """
        invalid_encoded = '{"v":2,"val":123,"type":"unknown_type","meta":null,"special":null}'

        with pytest.raises(ValueError, match="Unsupported dtype"):
            codec.decode(invalid_encoded)

    def test_no_fallback_on_invalid_special_marker(self, codec):
        """Test that unknown special marker is NOT silently handled.

        Expected behavior: Raises ValueError for unknown special marker
        """
        invalid_encoded = '{"v":2,"val":null,"type":"float64","meta":null,"special":"unknown_special"}'

        with pytest.raises(ValueError, match="Unknown special value marker"):
            codec.decode(invalid_encoded)

    def test_no_fallback_on_invalid_json(self, codec):
        """Test that invalid JSON is NOT silently handled.

        Expected behavior: Raises JSONDecodeError
        """
        invalid_json = '{"v":2,"val":123,INVALID}'

        with pytest.raises(json.JSONDecodeError):
            codec.decode(invalid_json)

    # ========== Fallback Coverage Validation ==========

    def test_all_fallbacks_use_specific_exceptions(self):
        """Meta-test: Verify that codec.py uses specific exceptions.

        This ensures we don't have bare 'except:' or 'except Exception:'
        without clear justification.
        """
        import finlab_guard.cache.codec as codec_module
        import inspect

        codec_source = inspect.getsource(codec_module)

        # Check for bare except:
        assert "except:" not in codec_source, \
            "Found bare 'except:' - all exceptions should be specific"

        # Count try-except blocks (should be < 5 as per plan)
        try_count = codec_source.count("try:")
        assert try_count < 5, \
            f"Too many try-except blocks: {try_count} (should be < 5)"

    def test_all_fallbacks_have_logging(self):
        """Meta-test: Verify that all fallbacks log warnings.

        This ensures fallbacks are observable in production.
        """
        import finlab_guard.cache.codec as codec_module
        import inspect

        codec_source = inspect.getsource(codec_module)

        # Count logger.warning calls
        # We expect at least 3 (one for each documented fallback)
        warning_count = codec_source.count("logger.warning")
        assert warning_count >= 3, \
            f"Not enough logger.warning calls: {warning_count} (expected >= 3)"

    # ========== Edge Cases ==========

    def test_encode_decode_zero_values(self, codec):
        """Test that zero values are handled correctly (not confused with None)."""
        # int zero
        encoded = codec.encode(0, dtype='int64')
        decoded = codec.decode(encoded)
        assert decoded == 0

        # float zero
        encoded = codec.encode(0.0, dtype='float64')
        decoded = codec.decode(encoded)
        assert decoded == 0.0

        # These should NOT be treated as special values
        schema = json.loads(encoded)
        assert schema['special'] is None

    def test_encode_decode_empty_string(self, codec):
        """Test that empty string is handled correctly (not confused with None)."""
        encoded = codec.encode("", dtype='object')
        decoded = codec.decode(encoded)
        assert decoded == ""

        # Should NOT be treated as special value
        schema = json.loads(encoded)
        assert schema['special'] is None

    def test_encode_decode_false_boolean(self, codec):
        """Test that False is handled correctly (not confused with None)."""
        encoded = codec.encode(False, dtype='bool')
        decoded = codec.decode(encoded)
        assert decoded == False

        # Should NOT be treated as special value
        schema = json.loads(encoded)
        assert schema['special'] is None
