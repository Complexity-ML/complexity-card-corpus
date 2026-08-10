from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Iterable


@dataclass(frozen=True)
class StaticTextProgress:
    """Count authored static and variable template lines in one Python source."""

    filename: str
    static_lines: int
    variable_lines: int

    @property
    def total_lines(self) -> int:
        return self.static_lines + self.variable_lines

    @property
    def static_ratio(self) -> float:
        return self.static_lines / self.total_lines if self.total_lines else 0.0

    @property
    def static_percent(self) -> float:
        """Percentage of measured authored text that remains fully static."""

        return self.static_ratio * 100.0


@dataclass(frozen=True)
class TemplateDensityProgress:
    """Measure literal prose versus semantic fields inside long templates."""

    filename: str
    literal_words: int
    variable_fields: int

    @property
    def static_percent(self) -> float:
        total = self.literal_words + self.variable_fields
        return self.literal_words / total * 100.0 if total else 0.0


def _docstring_nodes(tree: ast.AST) -> set[ast.Constant]:
    return {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }


def _format_field_count(value: str) -> int:
    """Count replacement fields used by ``str.format``/``format_map``."""

    try:
        return sum(
            field_name is not None
            for _literal, field_name, _format_spec, _conversion in Formatter().parse(
                value
            )
        )
    except ValueError:
        return 0


def _is_mapping_key(node: ast.Constant, parent: ast.AST | None) -> bool:
    """Return whether a string is a lookup key rather than rendered prose."""

    return isinstance(parent, ast.Dict) and node in parent.keys


def analyze_static_text(path: Path, *, minimum_words: int = 8) -> StaticTextProgress:
    """Measure long authored strings with and without interpolation fields."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    docstrings = _docstring_nodes(tree)
    static_lines: set[int] = set()
    variable_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            literal_words = sum(
                len(value.value.split())
                for value in node.values
                if isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            )
            fields = sum(
                isinstance(value, ast.FormattedValue) for value in node.values
            )
            if literal_words + fields >= minimum_words:
                variable_lines.add(node.lineno)
            continue
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node not in docstrings
            and not isinstance(parents.get(node), ast.JoinedStr)
            and not _is_mapping_key(node, parents.get(node))
        ):
            continue
        fields = _format_field_count(node.value)
        word_count = len(node.value.split())
        if word_count + fields < minimum_words:
            continue
        if fields:
            variable_lines.add(node.lineno)
        else:
            static_lines.add(node.lineno)
    static_only = static_lines - variable_lines
    return StaticTextProgress(
        filename=path.name,
        static_lines=len(static_only),
        variable_lines=len(variable_lines),
    )


def analyze_static_text_progress(
    paths: Iterable[Path], *, minimum_words: int = 8
) -> tuple[StaticTextProgress, ...]:
    return tuple(
        analyze_static_text(path, minimum_words=minimum_words)
        for path in paths
    )


def aggregate_static_text_progress(
    progress: Iterable[StaticTextProgress],
) -> StaticTextProgress:
    rows = tuple(progress)
    return StaticTextProgress(
        filename="TOTAL",
        static_lines=sum(row.static_lines for row in rows),
        variable_lines=sum(row.variable_lines for row in rows),
    )


def analyze_template_density(
    path: Path, *, minimum_units: int = 8
) -> TemplateDensityProgress:
    """Count literal words and replacement fields in authored long-form strings."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    docstrings = _docstring_nodes(tree)
    literal_words = 0
    variable_fields = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            words = sum(
                len(value.value.split())
                for value in node.values
                if isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            )
            fields = sum(
                isinstance(value, ast.FormattedValue) for value in node.values
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node not in docstrings
            and not isinstance(parents.get(node), ast.JoinedStr)
            and not _is_mapping_key(node, parents.get(node))
        ):
            parsed = tuple(Formatter().parse(node.value))
            words = sum(len(text.split()) for text, *_rest in parsed)
            fields = sum(field is not None for _text, field, *_rest in parsed)
        else:
            continue
        if words + fields >= minimum_units:
            literal_words += words
            variable_fields += fields
    return TemplateDensityProgress(path.name, literal_words, variable_fields)
