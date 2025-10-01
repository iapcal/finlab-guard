"""Custom exceptions for finlab-guard."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ..cache.manager import ChangeResult


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

    def __init__(self, message: str, changes: Union[list[Change], "ChangeResult"]):
        super().__init__(message)
        self.changes = changes

    def __str__(self) -> str:
        # Handle both old Change list format and new ChangeResult format
        if hasattr(self.changes, "cell_changes"):
            # New ChangeResult format
            from ..cache.manager import ChangeResult

            change_result = self.changes
            assert isinstance(change_result, ChangeResult)  # Type narrowing for mypy
            details = []

            # Add dtype changes (check this first as it's most significant)
            if hasattr(change_result, "dtype_changed") and change_result.dtype_changed:
                details.append("Dtype changes detected")

            # Add cell changes
            if not change_result.cell_changes.empty:
                cell_count = len(change_result.cell_changes)
                details.append(f"Cell modifications: {cell_count}")

            # Add row deletions
            if not change_result.row_deletions.empty:
                row_del_count = len(change_result.row_deletions)
                details.append(f"Row deletions: {row_del_count}")

            # Add column additions
            if not change_result.column_additions.empty:
                col_add_count = len(change_result.column_additions)
                details.append(f"Column additions: {col_add_count}")

            # Add column deletions
            if not change_result.column_deletions.empty:
                col_del_count = len(change_result.column_deletions)
                details.append(f"Column deletions: {col_del_count}")

            # Add row additions
            if not change_result.row_additions.empty:
                row_add_count = len(change_result.row_additions)
                details.append(f"Row additions: {row_add_count}")

            change_details = "\n".join(details) if details else "No detailed changes available"
        else:
            # Old Change list format (for backward compatibility)
            change_list = self.changes
            change_details = "\n".join(
                str(change) for change in change_list[:5]
            )  # Show first 5
            if len(change_list) > 5:
                change_details += f"\n... and {len(change_list) - 5} more changes"

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
