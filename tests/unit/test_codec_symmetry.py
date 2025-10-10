"""Symmetry tests for DataTypeCodec.

This test suite ensures that encode/decode are symmetric:
    decode(encode(x)) == x

For all supported data types and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from finlab_guard.cache.codec import DataTypeCodec


class TestCodecSymmetry:
    """Test encode/decode symmetry for all supported types."""

    @pytest.fixture
    def codec(self):
        """Create a DataTypeCodec instance."""
        return DataTypeCodec()

    # ========== Parameterized Integer Tests ==========

    @pytest.mark.parametrize("dtype,values", [
        ('int8', [np.int8(-128), np.int8(0), np.int8(127)]),
        ('int16', [np.int16(-32768), np.int16(0), np.int16(32767)]),
        ('int32', [np.int32(-2147483648), np.int32(0), np.int32(2147483647)]),
        ('int64', [np.int64(-9223372036854775808), np.int64(0), np.int64(9223372036854775807)]),
        ('uint8', [np.uint8(0), np.uint8(128), np.uint8(255)]),
        ('uint16', [np.uint16(0), np.uint16(32768), np.uint16(65535)]),
        ('uint32', [np.uint32(0), np.uint32(2147483648), np.uint32(4294967295)]),
        ('uint64', [np.uint64(0), np.uint64(9223372036854775808)]),
    ])
    def test_integer_symmetry(self, codec, dtype, values):
        """Test encode/decode symmetry for integer types.

        Verifies that decode(encode(x)) == x for all integer types and edge values.
        """
        for value in values:
            encoded = codec.encode(value, dtype=dtype)
            decoded = codec.decode(encoded)

            assert decoded == value, f"Symmetry failed for {dtype}: {value}"
            # Verify type is preserved
            assert decoded.dtype == np.dtype(dtype), \
                f"Type not preserved: expected {dtype}, got {decoded.dtype}"

    # ========== Parameterized Float Tests ==========

    @pytest.mark.parametrize("dtype,values", [
        ('float16', [np.float16(0.0), np.float16(1.5), np.float16(-1.5)]),
        ('float32', [np.float32(0.0), np.float32(3.14159), np.float32(-2.71828)]),
        ('float64', [np.float64(0.0), np.float64(3.141592653589793), np.float64(-2.718281828459045)]),
    ])
    def test_float_symmetry(self, codec, dtype, values):
        """Test encode/decode symmetry for float types."""
        for value in values:
            encoded = codec.encode(value, dtype=dtype)
            decoded = codec.decode(encoded)

            np.testing.assert_allclose(decoded, value, rtol=1e-6)
            assert decoded.dtype == np.dtype(dtype), \
                f"Type not preserved: expected {dtype}, got {decoded.dtype}"

    # ========== Parameterized Special Value Tests ==========

    @pytest.mark.parametrize("value,dtype", [
        (np.nan, 'float64'),
        (np.inf, 'float64'),
        (-np.inf, 'float64'),
        (pd.NA, 'object'),
        (None, 'object'),
    ])
    def test_special_value_symmetry(self, codec, value, dtype):
        """Test encode/decode symmetry for special values."""
        encoded = codec.encode(value, dtype=dtype)
        decoded = codec.decode(encoded)

        # Special handling for different special values
        if value is None:
            assert decoded is None
        elif pd.isna(value) and value is pd.NA:
            assert decoded is pd.NA
        elif isinstance(value, float) and np.isnan(value):
            assert np.isnan(decoded)
        elif isinstance(value, float) and np.isinf(value):
            assert np.isinf(decoded)
            assert (decoded > 0) == (value > 0)  # Sign preserved

    # ========== Boolean Symmetry ==========

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_bool_symmetry(self, codec, value):
        """Test encode/decode symmetry for boolean values."""
        encoded = codec.encode(value, dtype='bool')
        decoded = codec.decode(encoded)

        assert decoded == value
        assert isinstance(decoded, np.bool_)

    # ========== String Symmetry ==========

    @pytest.mark.parametrize("value", [
        "",
        "simple",
        "Hello, World!",
        "你好世界",
        "🌍🚀💡",
        "Line1\nLine2\nLine3",
        "Tab\tSeparated\tValues",
        '{"key": "value"}',  # JSON-like string
        "Special chars: @#$%^&*()",
    ])
    def test_string_symmetry(self, codec, value):
        """Test encode/decode symmetry for string values."""
        encoded = codec.encode(value, dtype='object')
        decoded = codec.decode(encoded)

        assert decoded == value
        assert isinstance(decoded, str)

    # ========== Datetime Symmetry ==========

    @pytest.mark.parametrize("value", [
        pd.Timestamp('2025-01-01'),
        pd.Timestamp('2025-10-10 12:34:56'),
        pd.Timestamp('1970-01-01 00:00:00'),
        pd.Timestamp('2262-01-01'),  # Near upper bound of datetime64[ns]
    ])
    def test_datetime_symmetry(self, codec, value):
        """Test encode/decode symmetry for datetime values."""
        encoded = codec.encode(value, dtype='datetime64[ns]')
        decoded = codec.decode(encoded)

        assert decoded == value
        assert isinstance(decoded, pd.Timestamp)

    # ========== Row Symmetry ==========

    def test_row_symmetry_simple(self, codec):
        """Test encode_row/decode_row symmetry for simple row."""
        original_row = {
            'int_col': 123,
            'float_col': 3.14,
            'str_col': "test",
            'bool_col': True
        }
        dtypes = {
            'int_col': 'int64',
            'float_col': 'float64',
            'str_col': 'object',
            'bool_col': 'bool'
        }

        encoded = codec.encode_row(original_row, dtypes=dtypes)
        decoded_row = codec.decode_row(encoded)

        assert set(decoded_row.keys()) == set(original_row.keys())
        assert int(decoded_row['int_col']) == original_row['int_col']
        np.testing.assert_allclose(decoded_row['float_col'], original_row['float_col'])
        assert decoded_row['str_col'] == original_row['str_col']
        assert decoded_row['bool_col'] == original_row['bool_col']

    def test_row_symmetry_with_special_values(self, codec):
        """Test row symmetry with special values."""
        original_row = {
            'normal': 100,
            'nan_val': np.nan,
            'none_val': None,
            'pd_na': pd.NA,
            'inf_val': np.inf
        }
        dtypes = {
            'normal': 'int64',
            'nan_val': 'float64',
            'none_val': 'object',
            'pd_na': 'object',
            'inf_val': 'float64'
        }

        encoded = codec.encode_row(original_row, dtypes=dtypes)
        decoded_row = codec.decode_row(encoded)

        assert decoded_row['normal'] == original_row['normal']
        assert np.isnan(decoded_row['nan_val'])
        assert decoded_row['none_val'] is None
        assert decoded_row['pd_na'] is pd.NA
        assert np.isinf(decoded_row['inf_val']) and decoded_row['inf_val'] > 0

    def test_row_symmetry_all_types(self, codec):
        """Test row symmetry with all supported types."""
        original_row = {
            'int8': np.int8(10),
            'int16': np.int16(1000),
            'int32': np.int32(100000),
            'int64': np.int64(10000000),
            'uint8': np.uint8(200),
            'float32': np.float32(3.14),
            'float64': np.float64(2.718),
            'bool': True,
            'str': "test",
            'datetime': pd.Timestamp('2025-10-10'),
            'nan': np.nan,
            'none': None,
        }
        dtypes = {
            'int8': 'int8',
            'int16': 'int16',
            'int32': 'int32',
            'int64': 'int64',
            'uint8': 'uint8',
            'float32': 'float32',
            'float64': 'float64',
            'bool': 'bool',
            'str': 'object',
            'datetime': 'datetime64[ns]',
            'nan': 'float64',
            'none': 'object',
        }

        encoded = codec.encode_row(original_row, dtypes=dtypes)
        decoded_row = codec.decode_row(encoded)

        # Verify all columns present
        assert set(decoded_row.keys()) == set(original_row.keys())

        # Verify each value
        assert decoded_row['int8'] == original_row['int8']
        assert decoded_row['int16'] == original_row['int16']
        assert decoded_row['int32'] == original_row['int32']
        assert decoded_row['int64'] == original_row['int64']
        assert decoded_row['uint8'] == original_row['uint8']
        np.testing.assert_allclose(decoded_row['float32'], original_row['float32'], rtol=1e-5)
        np.testing.assert_allclose(decoded_row['float64'], original_row['float64'])
        assert decoded_row['bool'] == original_row['bool']
        assert decoded_row['str'] == original_row['str']
        assert decoded_row['datetime'] == original_row['datetime']
        assert np.isnan(decoded_row['nan'])
        assert decoded_row['none'] is None

    # ========== Edge Case Symmetry ==========

    def test_symmetry_with_type_inference(self, codec):
        """Test symmetry when dtype is inferred (not explicitly provided)."""
        test_values = [
            (123, np.int64),
            (3.14, np.float64),
            (True, np.bool_),
            ("test", str),
        ]

        for value, expected_type in test_values:
            encoded = codec.encode(value)  # No explicit dtype
            decoded = codec.decode(encoded)

            # Value should match
            if isinstance(expected_type, type) and expected_type == str:
                assert decoded == value
            else:
                if isinstance(value, float):
                    np.testing.assert_allclose(decoded, value)
                else:
                    assert decoded == value

    def test_empty_row_symmetry(self, codec):
        """Test symmetry for empty row."""
        original_row = {}
        encoded = codec.encode_row(original_row)
        decoded_row = codec.decode_row(encoded)

        assert decoded_row == {}

    def test_large_integer_symmetry(self, codec):
        """Test symmetry for very large integers."""
        large_values = [
            (np.int64(9223372036854775807), 'int64'),  # Max int64
            (np.uint64(18446744073709551615), 'uint64'),  # Max uint64
        ]

        for value, dtype in large_values:
            encoded = codec.encode(value, dtype=dtype)
            decoded = codec.decode(encoded)

            assert decoded == value
            assert decoded.dtype == np.dtype(dtype)

    def test_precision_symmetry_float64(self, codec):
        """Test that float64 precision is fully preserved."""
        # Use a value with many decimal places
        value = np.float64(3.141592653589793238462643383279502884197)

        encoded = codec.encode(value, dtype='float64')
        decoded = codec.decode(encoded)

        # Should preserve full float64 precision
        np.testing.assert_equal(decoded, value)
