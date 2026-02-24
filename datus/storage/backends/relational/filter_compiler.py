# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""SQL filter compiler for parameterized queries.

This module compiles FilterExpr AST nodes into SQL WHERE clauses with
parameterized values to prevent SQL injection. It reuses the AST types
from datus.storage.lancedb_conditions.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, List, Optional, Tuple, Union

from datus.storage.lancedb_conditions import And, Condition, Not, Op, Or


class SQLFilterCompiler:
    """Compiles FilterExpr to parameterized SQL WHERE clauses.

    This compiler produces (where_clause, params) tuples suitable for use
    with parameterized SQL queries (using ? placeholders for SQLite).

    Example:
        >>> compiler = SQLFilterCompiler()
        >>> expr = And([Condition("status", Op.EQ, "active"), Condition("age", Op.GT, 18)])
        >>> clause, params = compiler.compile(expr)
        >>> print(clause)
        (status = ? AND age > ?)
        >>> print(params)
        ['active', 18]
    """

    def __init__(self, placeholder: str = "?"):
        """Initialize the compiler.

        Args:
            placeholder: Parameter placeholder string (? for SQLite, %s for PostgreSQL)
        """
        self.placeholder = placeholder

    def compile(self, expr: Optional[Union[Condition, And, Or, Not, str, Any]]) -> Tuple[Optional[str], List[Any]]:
        """Compile a FilterExpr to SQL WHERE clause with parameters.

        Args:
            expr: Filter expression to compile, or None

        Returns:
            Tuple of (where_clause, params). where_clause is None if expr is None.
        """
        if expr is None:
            return None, []

        # Handle raw string expressions (pass through, no params)
        if isinstance(expr, str):
            stripped = expr.strip()
            return (stripped or None, [])

        params: List[Any] = []
        clause = self._compile_node(expr, params)
        return clause, params

    def _compile_node(self, node: Union[Condition, And, Or, Not, Any], params: List[Any]) -> str:
        """Recursively compile an AST node.

        Args:
            node: AST node to compile
            params: List to append parameter values to

        Returns:
            SQL clause string
        """
        if isinstance(node, Condition):
            return self._compile_condition(node, params)
        if isinstance(node, And):
            return self._compile_and(node, params)
        if isinstance(node, Or):
            return self._compile_or(node, params)
        if isinstance(node, Not):
            return self._compile_not(node, params)

        # Fallback: try to use the node directly if it has the expected attributes
        # This supports FilterExpr from backends.interfaces which has the same structure
        if hasattr(node, "field") and hasattr(node, "op") and hasattr(node, "value"):
            return self._compile_condition(node, params)
        if hasattr(node, "nodes"):
            # Could be And or Or
            if node.__class__.__name__ == "And":
                return self._compile_and(node, params)
            if node.__class__.__name__ == "Or":
                return self._compile_or(node, params)
        if hasattr(node, "node"):
            if node.__class__.__name__ == "Not":
                return self._compile_not(node, params)

        raise TypeError(f"Unknown node type: {type(node)}")

    def _compile_condition(self, c: Any, params: List[Any]) -> str:
        """Compile a single condition to SQL.

        Args:
            c: Condition with field, op, and value attributes
            params: List to append parameter values to

        Returns:
            SQL condition string
        """
        field = self._escape_identifier(c.field)
        op = c.op
        value = c.value

        # Handle Op enum or string
        op_value = op.value if hasattr(op, "value") else str(op)

        # NULL handling
        if value is None:
            if op_value == "=" or op == Op.EQ:
                return f"{field} IS NULL"
            if op_value == "!=" or op == Op.NE:
                return f"{field} IS NOT NULL"
            raise ValueError(f"Operator {op} is invalid with NULL (field: {c.field})")

        # IN operator - expand to OR chain with parameters
        if op_value == "IN" or op == Op.IN:
            if not hasattr(value, "__iter__") or isinstance(value, (str, bytes)):
                raise TypeError("IN expects a non-string iterable value")

            values = list(value)
            if not values:
                return "1 = 0"  # Empty IN is always false

            non_null_values = [v for v in values if v is not None]
            include_null = any(v is None for v in values)

            parts = []
            for v in non_null_values:
                params.append(self._convert_value(v))
                parts.append(f"{field} = {self.placeholder}")

            if include_null:
                parts.append(f"{field} IS NULL")

            return "(" + " OR ".join(parts) + ")"

        # LIKE operator
        if op_value == "LIKE" or op == Op.LIKE:
            params.append(self._convert_value(value))
            return f"{field} LIKE {self.placeholder}"

        # Standard comparison operators
        if op_value in {"=", "!=", ">", ">=", "<", "<="} or op in {
            Op.EQ,
            Op.NE,
            Op.GT,
            Op.GTE,
            Op.LT,
            Op.LTE,
        }:
            params.append(self._convert_value(value))
            return f"{field} {op_value} {self.placeholder}"

        raise ValueError(f"Unsupported operator: {op}")

    def _compile_and(self, node: Any, params: List[Any]) -> str:
        """Compile AND node."""
        parts = []
        for n in node.nodes:
            if n is not None:
                parts.append(self._compile_node(n, params))

        if not parts:
            return "1 = 1"  # Empty AND is always true

        return "(" + " AND ".join(parts) + ")"

    def _compile_or(self, node: Any, params: List[Any]) -> str:
        """Compile OR node."""
        parts = []
        for n in node.nodes:
            if n is not None:
                parts.append(self._compile_node(n, params))

        if not parts:
            return "1 = 0"  # Empty OR is always false

        return "(" + " OR ".join(parts) + ")"

    def _compile_not(self, node: Any, params: List[Any]) -> str:
        """Compile NOT node."""
        inner = self._compile_node(node.node, params)
        return f"(NOT {inner})"

    def _escape_identifier(self, name: str) -> str:
        """Escape a SQL identifier (column/table name).

        Args:
            name: Identifier to escape

        Returns:
            Escaped identifier string
        """
        safe = name.strip()
        if not safe:
            raise ValueError("Identifier cannot be empty")

        # Check if quoting is needed
        first_char_requires_quote = not (safe[0].isalpha() or safe[0] == "_")
        needs_quote = first_char_requires_quote or any(c in safe for c in ' "().-+/\\|&*[]=<>!')

        if needs_quote:
            escaped = safe.replace('"', '""')
            return f'"{escaped}"'

        return safe

    def _convert_value(self, value: Any) -> Any:
        """Convert Python value to SQL parameter value.

        Args:
            value: Python value to convert

        Returns:
            Value suitable for SQL parameter binding
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float, str)):
            return value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        # Convert other types to string
        return str(value)


# Module-level singleton for convenience
_default_compiler = SQLFilterCompiler()


def compile_filter(expr: Optional[Union[Condition, And, Or, Not, str, Any]]) -> Tuple[Optional[str], List[Any]]:
    """Compile a FilterExpr to SQL WHERE clause with parameters.

    Convenience function using the default compiler.

    Args:
        expr: Filter expression to compile

    Returns:
        Tuple of (where_clause, params)

    Example:
        >>> from datus.storage.lancedb_conditions import eq, and_
        >>> clause, params = compile_filter(and_(eq("status", "active"), eq("age", 18)))
        >>> print(clause)
        (status = ? AND age = ?)
        >>> print(params)
        ['active', 18]
    """
    return _default_compiler.compile(expr)
