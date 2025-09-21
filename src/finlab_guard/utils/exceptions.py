"""Custom exceptions for finlab-guard."""

from datetime import datetime
from typing import Any


class Change:
    """Represents a data change at specific coordinates."""

    def __init__(
        self, coord: tuple, old_value: Any, new_value: Any, timestamp: datetime
    ):
        self.coord = coord
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"Change(coord={self.coord}, {self.old_value} -> {self.new_value})"


class DataModifiedException(Exception):
    """Raised when historical data has been modified."""

    def __init__(self, message: str, changes: list[Change]):
        super().__init__(message)
        self.changes = changes

    def __str__(self) -> str:
        change_details = "\n".join(
            str(change) for change in self.changes[:5]
        )  # Show first 5
        if len(self.changes) > 5:
            change_details += f"\n... and {len(self.changes) - 5} more changes"
        return f"{super().__str__()}\nChanges:\n{change_details}"


class FinlabConnectionException(Exception):
    """Raised when unable to connect to finlab."""

    pass


class UnsupportedDataFormatException(Exception):
    """Raised when encountering unsupported data format (e.g., MultiIndex columns)."""

    pass


class InvalidDataTypeException(Exception):
    """Raised when data type is invalid (e.g., not a DataFrame)."""

    pass
