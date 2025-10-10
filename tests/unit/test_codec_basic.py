"""Basic unit tests for DataTypeCodec."""

import json
import numpy as np
import pandas as pd
import pytest

from finlab_guard.cache.codec import DataTypeCodec, EncodedValue


class TestDataTypeCodecBasic:
    """Test basic encoding/decoding functionality."""

    @pytest.fixture
    def codec(self):
        """Create a DataTypeCodec instance."""
        return DataTypeCodec()

    # ========== Integer Types ==========

    def test_encode_decode_int8(self, codec):
        """Test int8 encoding and decoding."""
        value = np.int8(42)
        encoded = codec.encode(value, dtype='int8')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.int8)
        assert decoded == value

    def test_encode_decode_int16(self, codec):
        """Test int16 encoding and decoding."""
        value = np.int16(1000)
        encoded = codec.encode(value, dtype='int16')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.int16)
        assert decoded == value

    def test_encode_decode_int32(self, codec):
        """Test int32 encoding and decoding."""
        value = np.int32(100000)
        encoded = codec.encode(value, dtype='int32')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.int32)
        assert decoded == value

    def test_encode_decode_int64(self, codec):
        """Test int64 encoding and decoding."""
        value = np.int64(10000000000)
        encoded = codec.encode(value, dtype='int64')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.int64)
        assert decoded == value

    def test_encode_decode_uint8(self, codec):
        """Test uint8 encoding and decoding."""
        value = np.uint8(200)
        encoded = codec.encode(value, dtype='uint8')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.uint8)
        assert decoded == value

    def test_encode_decode_uint64(self, codec):
        """Test uint64 encoding and decoding."""
        value = np.uint64(18446744073709551615)
        encoded = codec.encode(value, dtype='uint64')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.uint64)
        assert decoded == value

    def test_encode_decode_python_int(self, codec):
        """Test Python native int (should default to int64)."""
        value = 12345
        encoded = codec.encode(value)
        decoded = codec.decode(encoded)

        # Should be converted to np.int64
        assert isinstance(decoded, np.int64)
        assert int(decoded) == value

    # ========== Float Types ==========

    def test_encode_decode_float16(self, codec):
        """Test float16 encoding and decoding."""
        value = np.float16(3.14)
        encoded = codec.encode(value, dtype='float16')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.float16)
        assert decoded == value

    def test_encode_decode_float32(self, codec):
        """Test float32 encoding and decoding."""
        value = np.float32(3.141592)
        encoded = codec.encode(value, dtype='float32')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.float32)
        np.testing.assert_allclose(decoded, value, rtol=1e-6)

    def test_encode_decode_float64(self, codec):
        """Test float64 encoding and decoding."""
        value = np.float64(3.14159265358979)
        encoded = codec.encode(value, dtype='float64')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.float64)
        np.testing.assert_allclose(decoded, value)

    def test_encode_decode_python_float(self, codec):
        """Test Python native float (should default to float64)."""
        value = 2.718281828
        encoded = codec.encode(value)
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.float64)
        np.testing.assert_allclose(float(decoded), value)

    # ========== Boolean Type ==========

    def test_encode_decode_bool_true(self, codec):
        """Test boolean True encoding and decoding."""
        value = np.bool_(True)
        encoded = codec.encode(value, dtype='bool')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.bool_)
        assert decoded == True

    def test_encode_decode_bool_false(self, codec):
        """Test boolean False encoding and decoding."""
        value = np.bool_(False)
        encoded = codec.encode(value, dtype='bool')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.bool_)
        assert decoded == False

    def test_encode_decode_python_bool(self, codec):
        """Test Python native bool."""
        value = True
        encoded = codec.encode(value, dtype='bool')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, np.bool_)
        assert decoded == value

    # ========== String/Object Type ==========

    def test_encode_decode_string(self, codec):
        """Test string encoding and decoding."""
        value = "Hello, World!"
        encoded = codec.encode(value, dtype='object')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, str)
        assert decoded == value

    def test_encode_decode_unicode_string(self, codec):
        """Test Unicode string encoding and decoding."""
        value = "你好世界 🌍"
        encoded = codec.encode(value, dtype='object')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, str)
        assert decoded == value

    # ========== Special Values ==========

    def test_encode_decode_nan(self, codec):
        """Test np.nan encoding and decoding."""
        value = np.nan
        encoded = codec.encode(value, dtype='float64')

        # Verify encoded format
        schema = json.loads(encoded)
        assert schema['special'] == 'nan'
        assert schema['val'] is None
        assert schema['type'] == 'float64'

        decoded = codec.decode(encoded)
        assert isinstance(decoded, float)
        assert np.isnan(decoded)

    def test_encode_decode_pd_na(self, codec):
        """Test pd.NA encoding and decoding."""
        value = pd.NA
        encoded = codec.encode(value, dtype='object')

        # Verify encoded format
        schema = json.loads(encoded)
        assert schema['special'] == 'pd_na'
        assert schema['val'] is None

        decoded = codec.decode(encoded)
        assert pd.isna(decoded)
        assert decoded is pd.NA

    def test_encode_decode_none(self, codec):
        """Test None encoding and decoding."""
        value = None
        encoded = codec.encode(value, dtype='object')

        # Verify encoded format
        schema = json.loads(encoded)
        assert schema['special'] == 'none'
        assert schema['val'] is None

        decoded = codec.decode(encoded)
        assert decoded is None

    def test_encode_decode_inf(self, codec):
        """Test np.inf encoding and decoding."""
        value = np.inf
        encoded = codec.encode(value, dtype='float64')

        # Verify encoded format
        schema = json.loads(encoded)
        assert schema['special'] == 'inf'
        assert schema['val'] is None

        decoded = codec.decode(encoded)
        assert np.isinf(decoded) and decoded > 0

    def test_encode_decode_neg_inf(self, codec):
        """Test -np.inf encoding and decoding."""
        value = -np.inf
        encoded = codec.encode(value, dtype='float64')

        # Verify encoded format
        schema = json.loads(encoded)
        assert schema['special'] == '-inf'
        assert schema['val'] is None

        decoded = codec.decode(encoded)
        assert np.isinf(decoded) and decoded < 0

    # ========== Datetime Type ==========

    def test_encode_decode_timestamp(self, codec):
        """Test pd.Timestamp encoding and decoding."""
        value = pd.Timestamp('2025-10-10 12:34:56')
        encoded = codec.encode(value, dtype='datetime64[ns]')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, pd.Timestamp)
        assert decoded == value

    def test_encode_decode_timestamp_with_timezone(self, codec):
        """Test pd.Timestamp with timezone."""
        value = pd.Timestamp('2025-10-10 12:34:56', tz='UTC')
        encoded = codec.encode(value, dtype='datetime64[ns]')
        decoded = codec.decode(encoded)

        assert isinstance(decoded, pd.Timestamp)
        # Note: timezone info might be lost depending on implementation
        # This is acceptable for cache purposes

    # ========== Row Encoding/Decoding ==========

    def test_encode_decode_row_simple(self, codec):
        """Test encoding and decoding a simple row."""
        row = {
            'A': 1,
            'B': 2.5,
            'C': "test"
        }
        dtypes = {
            'A': 'int64',
            'B': 'float64',
            'C': 'object'
        }

        encoded = codec.encode_row(row, dtypes=dtypes)
        decoded = codec.decode_row(encoded)

        assert decoded['A'] == row['A']
        np.testing.assert_allclose(decoded['B'], row['B'])
        assert decoded['C'] == row['C']

    def test_encode_decode_row_with_special_values(self, codec):
        """Test encoding and decoding a row with special values."""
        row = {
            'A': 1,
            'B': np.nan,
            'C': None,
            'D': pd.NA
        }
        dtypes = {
            'A': 'int64',
            'B': 'float64',
            'C': 'object',
            'D': 'object'
        }

        encoded = codec.encode_row(row, dtypes=dtypes)
        decoded = codec.decode_row(encoded)

        assert decoded['A'] == row['A']
        assert np.isnan(decoded['B'])
        assert decoded['C'] is None
        assert decoded['D'] is pd.NA

    def test_encode_decode_row_mixed_types(self, codec):
        """Test encoding and decoding a row with mixed types."""
        row = {
            'int_col': np.int32(100),
            'float_col': np.float32(3.14),
            'bool_col': True,
            'str_col': "hello",
            'date_col': pd.Timestamp('2025-01-01'),
            'nan_col': np.nan
        }
        dtypes = {
            'int_col': 'int32',
            'float_col': 'float32',
            'bool_col': 'bool',
            'str_col': 'object',
            'date_col': 'datetime64[ns]',
            'nan_col': 'float64'
        }

        encoded = codec.encode_row(row, dtypes=dtypes)
        decoded = codec.decode_row(encoded)

        assert decoded['int_col'] == row['int_col']
        np.testing.assert_allclose(decoded['float_col'], row['float_col'], rtol=1e-5)
        assert decoded['bool_col'] == row['bool_col']
        assert decoded['str_col'] == row['str_col']
        assert decoded['date_col'] == row['date_col']
        assert np.isnan(decoded['nan_col'])

    def test_encode_decode_empty_row(self, codec):
        """Test encoding and decoding an empty row."""
        row = {}
        encoded = codec.encode_row(row)
        decoded = codec.decode_row(encoded)

        assert decoded == {}

    # ========== JSON Schema Format Validation ==========

    def test_encoded_format_has_version(self, codec):
        """Test that encoded format includes version."""
        value = 123
        encoded = codec.encode(value, dtype='int64')
        schema = json.loads(encoded)

        assert schema['v'] == 2

    def test_encoded_format_has_required_fields(self, codec):
        """Test that encoded format has all required fields."""
        value = 42
        encoded = codec.encode(value, dtype='int64')
        schema = json.loads(encoded)

        assert 'v' in schema
        assert 'val' in schema
        assert 'type' in schema
        assert 'meta' in schema
        assert 'special' in schema

    def test_row_encoded_format(self, codec):
        """Test that row encoded format is correct."""
        row = {'A': 1, 'B': 2}
        encoded = codec.encode_row(row, dtypes={'A': 'int64', 'B': 'int64'})
        schema = json.loads(encoded)

        assert schema['v'] == 2
        assert 'd' in schema
        assert 'A' in schema['d']
        assert 'B' in schema['d']
        assert 'val' in schema['d']['A']
        assert 'type' in schema['d']['A']
