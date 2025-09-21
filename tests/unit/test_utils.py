"""Unit tests for utils module."""

from datetime import datetime

import pytest

from finlab_guard.utils.exceptions import (
    Change,
    DataModifiedException,
    FinlabConnectionException,
    InvalidDataTypeException,
    UnsupportedDataFormatException,
)


class TestChange:
    """Test suite for Change class."""

    def test_change_creation(self):
        """Test creating a Change object."""
        coord = ("A", "col1")
        old_value = 1
        new_value = 2
        timestamp = datetime.now()

        change = Change(coord, old_value, new_value, timestamp)

        assert change.coord == coord
        assert change.old_value == old_value
        assert change.new_value == new_value
        assert change.timestamp == timestamp

    def test_change_repr(self):
        """Test Change object string representation."""
        coord = ("A", "col1")
        old_value = 1
        new_value = 2
        timestamp = datetime.now()

        change = Change(coord, old_value, new_value, timestamp)
        repr_str = repr(change)

        assert "Change(coord=('A', 'col1')" in repr_str
        assert "1 -> 2" in repr_str

    def test_change_addition(self):
        """Test Change object for addition (old_value is None)."""
        coord = ("B", "col2")
        old_value = None
        new_value = 5
        timestamp = datetime.now()

        change = Change(coord, old_value, new_value, timestamp)

        assert change.old_value is None
        assert change.new_value == 5

        repr_str = repr(change)
        assert "None -> 5" in repr_str

    def test_change_with_various_data_types(self):
        """Test Change object with various data types."""
        # String values
        change1 = Change(("A", "name"), "old", "new", datetime.now())
        assert change1.old_value == "old"
        assert change1.new_value == "new"

        # Float values
        change2 = Change(("B", "price"), 1.5, 2.7, datetime.now())
        assert change2.old_value == 1.5
        assert change2.new_value == 2.7

        # Boolean values
        change3 = Change(("C", "flag"), True, False, datetime.now())
        assert change3.old_value is True
        assert change3.new_value is False


class TestDataModifiedException:
    """Test suite for DataModifiedException class."""

    def test_data_modified_exception_creation(self):
        """Test creating DataModifiedException."""
        message = "Historical data has been modified"
        changes = [
            Change(("A", "col1"), 1, 2, datetime.now()),
            Change(("B", "col2"), 3, 4, datetime.now()),
        ]

        exception = DataModifiedException(message, changes)

        assert str(exception).startswith(message)
        assert exception.changes == changes
        assert len(exception.changes) == 2

    def test_data_modified_exception_str_few_changes(self):
        """Test string representation with few changes."""
        message = "Data modified"
        changes = [
            Change(("A", "col1"), 1, 2, datetime.now()),
            Change(("B", "col2"), 3, 4, datetime.now()),
        ]

        exception = DataModifiedException(message, changes)
        str_repr = str(exception)

        assert message in str_repr
        assert "Changes:" in str_repr
        assert "Change(coord=('A', 'col1')" in str_repr
        assert "Change(coord=('B', 'col2')" in str_repr

    def test_data_modified_exception_str_many_changes(self):
        """Test string representation with many changes (truncation)."""
        message = "Data modified"
        changes = []

        # Create 10 changes (more than the 5 displayed)
        for i in range(10):
            changes.append(Change((f"row_{i}", "col"), i, i + 1, datetime.now()))

        exception = DataModifiedException(message, changes)
        str_repr = str(exception)

        assert message in str_repr
        assert "Changes:" in str_repr
        assert "... and 5 more changes" in str_repr

        # Should only show first 5 changes
        assert "row_0" in str_repr
        assert "row_4" in str_repr
        assert "row_5" not in str_repr  # Should be truncated

    def test_data_modified_exception_empty_changes(self):
        """Test DataModifiedException with empty changes."""
        message = "Data modified"
        changes = []

        exception = DataModifiedException(message, changes)
        str_repr = str(exception)

        assert message in str_repr
        assert "Changes:" in str_repr

    def test_data_modified_exception_inheritance(self):
        """Test that DataModifiedException inherits from Exception."""
        changes = []
        exception = DataModifiedException("test", changes)

        assert isinstance(exception, Exception)


class TestFinlabConnectionException:
    """Test suite for FinlabConnectionException class."""

    def test_finlab_connection_exception_creation(self):
        """Test creating FinlabConnectionException."""
        message = "Cannot connect to finlab"
        exception = FinlabConnectionException(message)

        assert str(exception) == message

    def test_finlab_connection_exception_inheritance(self):
        """Test that FinlabConnectionException inherits from Exception."""
        exception = FinlabConnectionException("test")
        assert isinstance(exception, Exception)

    def test_finlab_connection_exception_chaining(self):
        """Test exception chaining."""
        original_error = ConnectionError("Network timeout")

        try:
            try:
                raise original_error
            except ConnectionError as e:
                raise FinlabConnectionException("Cannot fetch data") from e
        except FinlabConnectionException as chained_exception:
            assert chained_exception.__cause__ == original_error


class TestUnsupportedDataFormatException:
    """Test suite for UnsupportedDataFormatException class."""

    def test_unsupported_data_format_exception_creation(self):
        """Test creating UnsupportedDataFormatException."""
        message = "MultiIndex columns are not supported"
        exception = UnsupportedDataFormatException(message)

        assert str(exception) == message

    def test_unsupported_data_format_exception_inheritance(self):
        """Test that UnsupportedDataFormatException inherits from Exception."""
        exception = UnsupportedDataFormatException("test")
        assert isinstance(exception, Exception)


class TestInvalidDataTypeException:
    """Test suite for InvalidDataTypeException class."""

    def test_invalid_data_type_exception_creation(self):
        """Test creating InvalidDataTypeException."""
        message = "Expected DataFrame, got str"
        exception = InvalidDataTypeException(message)

        assert str(exception) == message

    def test_invalid_data_type_exception_inheritance(self):
        """Test that InvalidDataTypeException inherits from Exception."""
        exception = InvalidDataTypeException("test")
        assert isinstance(exception, Exception)


class TestExceptionIntegration:
    """Test suite for exception integration scenarios."""

    def test_exception_usage_in_try_catch(self):
        """Test using exceptions in try-catch blocks."""
        changes = [Change(("A", "col1"), 1, 2, datetime.now())]

        # Test DataModifiedException
        with pytest.raises(DataModifiedException) as exc_info:
            raise DataModifiedException("Data changed", changes)

        caught_exception = exc_info.value
        assert len(caught_exception.changes) == 1
        assert caught_exception.changes[0].coord == ("A", "col1")

        # Test FinlabConnectionException
        with pytest.raises(FinlabConnectionException):
            raise FinlabConnectionException("Connection failed")

        # Test UnsupportedDataFormatException
        with pytest.raises(UnsupportedDataFormatException):
            raise UnsupportedDataFormatException("Format not supported")

        # Test InvalidDataTypeException
        with pytest.raises(InvalidDataTypeException):
            raise InvalidDataTypeException("Invalid type")

    def test_exception_hierarchy(self):
        """Test that all custom exceptions inherit from Exception."""
        exceptions_to_test = [
            DataModifiedException("test", []),
            FinlabConnectionException("test"),
            UnsupportedDataFormatException("test"),
            InvalidDataTypeException("test"),
        ]

        for exception in exceptions_to_test:
            assert isinstance(exception, Exception)

    def test_change_in_exception_context(self):
        """Test Change objects in exception context."""
        timestamp = datetime.now()

        # Create changes with different scenarios
        modification = Change(("A", "col1"), 100, 105, timestamp)
        addition = Change(("B", "col2"), None, 200, timestamp)

        changes = [modification, addition]
        exception = DataModifiedException("Multiple changes detected", changes)

        # Verify changes are preserved correctly
        assert len(exception.changes) == 2

        # Check modification
        assert exception.changes[0].old_value == 100
        assert exception.changes[0].new_value == 105

        # Check addition
        assert exception.changes[1].old_value is None
        assert exception.changes[1].new_value == 200

    def test_exception_messages_formatting(self):
        """Test that exception messages are properly formatted."""
        # Test with special characters
        message_with_quotes = 'Data for "price:收盤價" has been modified'
        exception = DataModifiedException(message_with_quotes, [])
        str_repr = str(exception)
        assert message_with_quotes in str_repr
        assert "Changes:" in str_repr

        # Test with unicode
        unicode_message = "資料已被修改"
        exception = FinlabConnectionException(unicode_message)
        assert str(exception) == unicode_message

    def test_large_number_of_changes(self):
        """Test performance with large number of changes."""
        timestamp = datetime.now()

        # Create a large number of changes
        changes = []
        for i in range(1000):
            change = Change((f"row_{i}", "col"), i, i + 1, timestamp)
            changes.append(change)

        exception = DataModifiedException("Many changes", changes)

        # Should handle large number of changes
        assert len(exception.changes) == 1000

        # String representation should still be manageable (truncated)
        str_repr = str(exception)
        assert "... and 995 more changes" in str_repr
