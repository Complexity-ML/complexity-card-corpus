from __future__ import annotations

import hashlib
from string import Formatter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias


VariableCards: TypeAlias = tuple[str, ...]
VariableSenseTable: TypeAlias = Mapping[str, VariableCards]
VariableByTable: TypeAlias = Mapping[str, VariableSenseTable]
DealtVariableBy: TypeAlias = dict[str, dict[str, str]]


def _stable_index(key: str, size: int) -> int:
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return value % size


@dataclass(frozen=True)
class VariableBy2D:
    """A two-dimensional deck indexed first by axis, then semantic sense.

    Templates address dealt cells with Python's mapping syntax, for example
    ``{for[purpose]}`` or ``{audience[common_noun]}``. Each cell contains only
    mutually substitutable surfaces for that exact sense.
    """

    table: VariableByTable

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError("variable_by requires at least one axis")
        normalized: dict[str, Mapping[str, VariableCards]] = {}
        for axis, senses in self.table.items():
            if not axis.strip():
                raise ValueError("variable_by axis names must be visible")
            if not senses:
                raise ValueError(f"variable_by axis {axis!r} requires a sense")
            normalized_senses: dict[str, VariableCards] = {}
            for sense, cards in senses.items():
                if not sense.strip():
                    raise ValueError("variable_by sense names must be visible")
                normalized_cards = tuple(cards)
                if not normalized_cards or not all(
                    card.strip() for card in normalized_cards
                ):
                    raise ValueError(
                        f"variable_by cell {axis}[{sense}] requires visible cards"
                    )
                if len(set(normalized_cards)) != len(normalized_cards):
                    raise ValueError(
                        f"variable_by cell {axis}[{sense}] contains duplicate cards"
                    )
                normalized_senses[sense] = normalized_cards
            normalized[axis] = MappingProxyType(normalized_senses)
        object.__setattr__(self, "table", MappingProxyType(normalized))

    def field_names(self) -> tuple[str, ...]:
        return tuple(
            f"{axis}[{sense}]"
            for axis, senses in self.table.items()
            for sense in senses
        )

    def cards(self, axis: str, sense: str) -> VariableCards:
        try:
            return self.table[axis][sense]
        except KeyError as error:
            raise KeyError(f"unknown variable_by cell {axis}[{sense}]") from error

    def variable_for(self, axis: str, sense: str) -> VariableCards:
        """Return the synonym deck allowed for one axis in one exact sense."""

        return self.cards(axis, sense)

    @staticmethod
    def template_fields(template: str) -> tuple[str, ...]:
        """Return the ordered 2D cells requested by one text template."""

        try:
            fields = tuple(
                field_name
                for _literal, field_name, _format_spec, _conversion in (
                    Formatter().parse(template)
                )
                if field_name is not None
            )
        except ValueError as error:
            raise ValueError(f"invalid variable_by template: {template!r}") from error
        return tuple(dict.fromkeys(fields))

    def validate_templates(self, templates: tuple[str, ...]) -> tuple[str, ...]:
        """Validate templates and return every referenced cell in stable order."""

        requested = tuple(
            dict.fromkeys(
                field
                for template in templates
                for field in self.template_fields(template)
            )
        )
        available = set(self.field_names())
        unknown = tuple(field for field in requested if field not in available)
        if unknown:
            raise ValueError(
                "variable_by template references unknown cells: "
                + ", ".join(unknown)
            )
        return requested

    def render(self, template: str, dealt: DealtVariableBy) -> str:
        """Render one validated template from a previously dealt matrix."""

        self.validate_templates((template,))
        return template.format_map(dealt)

    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """Expose nested variable references for compatibility and cycle audits."""

        return {
            f"{axis}[{sense}]": tuple(
                dict.fromkeys(
                    field
                    for card in cards
                    for field in self.template_fields(card)
                )
            )
            for axis, senses in self.table.items()
            for sense, cards in senses.items()
        }

    def expand_dependencies(self, fields: tuple[str, ...]) -> tuple[str, ...]:
        """Expand requested cells to their ordered transitive dependencies."""

        graph = self.dependency_graph()
        expanded: list[str] = []

        def visit(field: str) -> None:
            if field in expanded:
                return
            expanded.append(field)
            for dependency in graph.get(field, ()):
                visit(dependency)

        for field in fields:
            visit(field)
        return tuple(expanded)

    def deal_indices(self, seed: str) -> dict[str, dict[str, int]]:
        """Return the deterministic card index selected in every 2D cell."""

        return {
            axis: {
                sense: _stable_index(f"{seed}:{axis}:{sense}", len(cards))
                for sense, cards in senses.items()
            }
            for axis, senses in self.table.items()
        }

    def deal(self, seed: str) -> DealtVariableBy:
        indices = self.deal_indices(seed)
        selected = {
            axis: {
                sense: cards[indices[axis][sense]]
                for sense, cards in senses.items()
            }
            for axis, senses in self.table.items()
        }
        available = set(self.field_names())
        unknown = {
            dependency
            for dependencies in self.dependency_graph().values()
            for dependency in dependencies
            if dependency not in available
        }
        if unknown:
            raise ValueError(
                "variable_by reservoir references unknown cells: "
                + ", ".join(sorted(unknown))
            )

        resolved: DealtVariableBy = {axis: {} for axis in selected}
        resolving: list[str] = []

        class LazyAxis(dict[str, str]):
            def __init__(self, axis: str) -> None:
                self.axis = axis

            def __getitem__(self, sense: str) -> str:
                return resolve(self.axis, sense)

        lazy = {axis: LazyAxis(axis) for axis in selected}

        def resolve(axis: str, sense: str) -> str:
            field = f"{axis}[{sense}]"
            if sense in resolved[axis]:
                return resolved[axis][sense]
            if field in resolving:
                cycle = " -> ".join((*resolving, field))
                raise ValueError(f"cyclic variable_by dependency: {cycle}")
            resolving.append(field)
            value = selected[axis][sense].format_map(lazy)
            resolving.pop()
            resolved[axis][sense] = value
            return value

        for axis, senses in selected.items():
            for sense in senses:
                resolve(axis, sense)
        return resolved
