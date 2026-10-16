"""Typed skill documents and deterministic, bounded semantic deltas.

Required wording is editable only when it is an exact rendering of the closed
obligation vocabulary below. This is a deliberately finite guarantee, not a
general natural-language equivalence checker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
from itertools import product
import json
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .contract import SkillContract, validate_skill_text


# Each atom has a canonical rendering followed by vetted, equivalent renderings.
SEMANTIC_ATOMS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "check_visible_state": (
        "Check the visible state before acting.",
        "Before acting, inspect the visible state.",
    ),
    "use_admissible_actions": (
        "Use only currently admissible actions.",
        "Choose only an action currently available.",
    ),
    "require_known_preconditions": (
        "Proceed only when the required preconditions are known.",
        "Do not proceed until every required precondition is known.",
    ),
    "verify_postcondition": (
        "Verify the postcondition from the visible state.",
        "Check the visible state to confirm the postcondition.",
    ),
    "verify_environment_completion": (
        "Stop only after the environment confirms completion.",
        "Continue until the environment confirms the task is complete.",
    ),
    "replan_if_action_unavailable": (
        "Replan if a required action is unavailable.",
        "If a required action is unavailable, make a new plan.",
    ),
    "acquire_before_transform": (
        "Acquire the item before transforming it.",
        "Transform the item only after acquiring it.",
    ),
})

_VALID_SECTIONS = {"applicability", "invariants", "procedure", "heuristics", "recovery", "verification", "anti_patterns", "preconditions", "postconditions"}
_VALID_KINDS = {"invariant", "procedure", "heuristic", "recovery", "verification", "anti_pattern", "precondition", "postcondition", "scope"}
_VALID_STATUSES = {"required", "recommended"}
_INSTANCE = re.compile(r"(?i)\b(?:cabinet|drawer|countertop|table|sinkbasin|stoveburner|fridge|microwave|bed|sofa)\s+\d+\b")


@dataclass(frozen=True)
class SemanticItem:
    item_id: str
    section: str
    kind: str
    status: str
    protected: bool
    semantic_tags: tuple[str, ...]
    evidence: tuple[str, ...]
    text: str
    semantic_atoms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) for value in (self.item_id, self.section, self.kind, self.status, self.text)):
            raise ValueError("semantic item text and identifiers must be strings")
        if not self.item_id.strip() or self.section not in _VALID_SECTIONS or self.kind not in _VALID_KINDS:
            raise ValueError("semantic item identity, section, or kind is invalid")
        if self.status not in _VALID_STATUSES or type(self.protected) is not bool or not self.text.strip():
            raise ValueError("semantic item status and text are required")
        for name in ("semantic_tags", "evidence", "semantic_atoms"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"semantic item {name} must be a tuple of strings")
        if len(set(self.semantic_atoms)) != len(self.semantic_atoms):
            raise ValueError("semantic atoms must be unique within an item")
        if any(atom not in SEMANTIC_ATOMS for atom in self.semantic_atoms):
            raise ValueError("unknown semantic atom")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticDocument:
    contract: SkillContract
    items: tuple[SemanticItem, ...]
    refinement_rounds: int = 0
    origin: str = "legacy_contract_conversion"
    source_body: str | None = None
    token_counter: Callable[[str], int] | None = field(default=None, repr=False, compare=False)
    revision_sources: tuple[str, ...] = ()
    parent_contract_digest: str | None = None
    parent_contract_version: int | None = None

    def __post_init__(self) -> None:
        ids = [item.item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic item IDs must be unique")
        if self.refinement_rounds < 0 or self.refinement_rounds > 2:
            raise ValueError("semantic refinements are limited to two rounds")
        if self.token_counter is None:
            raise ValueError("semantic documents require an explicit tokenizer/count function")
        if self.source_body is not None and self.refinement_rounds == 0:
            if self.source_body != self.contract.body or tuple(i.text for i in self.items if i.section == "procedure") != self.contract.procedure:
                raise ValueError("legacy source body differs from typed items")
        for item in self.items:
            if item.status == "required" and item.semantic_atoms:
                _validate_required_rendering(item)
        if self.token_count > 2000:
            raise ValueError("rendered semantic document exceeds 2000 counted tokens")

    @property
    def rendered(self) -> str:
        if self.refinement_rounds == 0 and self.source_body is not None:
            return self.source_body
        return self.to_contract().body

    @property
    def token_count(self) -> int:
        count = self.token_counter(self.rendered)
        if type(count) is not int or count < 0:
            raise ValueError("token counter must return a non-negative integer")
        return count

    @property
    def digest(self) -> str:
        payload = {
            "contract": self.contract.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "refinement_rounds": self.refinement_rounds,
            "origin": self.origin,
            "source_body": self.source_body,
            "token_count": self.token_count,
            "revision_sources": self.revision_sources,
            "parent_contract_digest": self.parent_contract_digest,
            "parent_contract_version": self.parent_contract_version,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_contract(self) -> SkillContract:
        if not any(i.section == "procedure" for i in self.items):
            raise ValueError("semantic document must retain a procedure")
        if not self.refinement_rounds and self.source_body is not None:
            return self.contract
        # The exported runtime body is the same body whose tokens/invariants were checked.
        procedure = tuple(i for section in ("procedure", "heuristics", "recovery", "verification", "anti_patterns")
                          for i in self.items if i.section == section and i.item_id != "legacy-stop-replan")
        ids = tuple(item.item_id for item in procedure)
        revision_metadata = ({
            "generator_provider": None,
            "model_identity": None,
            "proposal_prompt_sha256": None,
            "proposal_output_sha256": None,
        } if self.refinement_rounds else {})
        return replace(
            self.contract,
            version=self.contract.version + self.refinement_rounds,
            procedure=tuple(item.text if item.section == "procedure" else f"[{item.section}] {item.text}" for item in procedure),
            procedure_step_ids=ids,
            required_procedure_step_ids=tuple(item.item_id for item in procedure if item.status == "required"),
            parent_digest=self.parent_contract_digest if self.refinement_rounds else self.contract.parent_digest,
            parent_version=self.parent_contract_version if self.refinement_rounds else self.contract.parent_version,
            **revision_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"contract": self.contract.to_dict(), "items": [i.to_dict() for i in self.items],
                "refinement_rounds": self.refinement_rounds, "origin": self.origin,
                "source_body": self.source_body, "revision_sources": list(self.revision_sources),
                "parent_contract_digest": self.parent_contract_digest,
                "parent_contract_version": self.parent_contract_version, "digest": self.digest,
                "token_count": self.token_count}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, token_counter: Callable[[str], int]) -> "SemanticDocument":
        data = {k:v for k,v in value.items() if k not in {"digest", "token_count"}}
        data["contract"] = SkillContract.from_dict(data["contract"])
        data["items"] = tuple(_item_from_dict(i) for i in data["items"])
        data["revision_sources"] = tuple(data["revision_sources"])
        result = cls(**data,token_counter=token_counter)
        if result.digest != value["digest"] or result.token_count != value["token_count"]:
            raise ValueError("semantic document or tokenizer identity changed on readback")
        return result


@dataclass(frozen=True)
class SemanticOperation:
    op: str
    item_id: str | None = None
    item_ids: tuple[str, ...] = ()
    item: SemanticItem | None = None
    new_items: tuple[SemanticItem, ...] = ()
    after_id: str | None = None
    section: str | None = None
    order: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticOperation":
        if not isinstance(value, Mapping):
            raise ValueError("semantic operation must be an object")
        op = value.get("op")
        fields = {
            "add": {"op", "item", "after_id"},
            "modify": {"op", "item_id", "item"},
            "split": {"op", "item_id", "new_items"},
            "merge": {"op", "item_ids", "item"},
            "delete": {"op", "item_id"},
            "reorder": {"op", "section", "order"},
        }
        if op not in fields or set(value) - fields[op]:
            raise ValueError("semantic operation has an unknown operation or unexpected fields")
        for name in ("item_id", "after_id", "section"):
            if name in value and value[name] is not None and not isinstance(value[name], str):
                raise ValueError(f"semantic {name} must be a string")
        item = _item_from_dict(value.get("item")) if "item" in value else None
        raw_new_items = value.get("new_items", ())
        if not isinstance(raw_new_items, (list, tuple)):
            raise ValueError("split new_items must be a list")
        new_items = tuple(_item_from_dict(i) for i in raw_new_items)
        item_ids = _string_tuple(value.get("item_ids", ()), "item_ids")
        order = _string_tuple(value.get("order", ()), "order")
        return cls(
            op=op,
            item_id=value.get("item_id"),
            item_ids=item_ids,
            item=item,
            new_items=new_items,
            after_id=value.get("after_id"),
            section=value.get("section"),
            order=order,
        )


@dataclass(frozen=True)
class SemanticDelta:
    base_digest: str
    operations: tuple[SemanticOperation, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticDelta":
        if (not isinstance(value, Mapping) or set(value) != {"base_digest", "operations"}
                or not isinstance(value.get("base_digest"), str) or not isinstance(value.get("operations"), list)):
            raise ValueError("semantic delta must contain only base_digest and operations")
        return cls(value["base_digest"], tuple(SemanticOperation.from_dict(op) for op in value["operations"]))


def document_from_contract(
    contract: SkillContract,
    *,
    token_counter: Callable[[str], int],
    semantic_atom_bindings: Mapping[str, Sequence[str]] | None = None,
) -> SemanticDocument:
    """Convert legacy data literally; optional atom bindings are explicit, never inferred."""
    items: list[SemanticItem] = []
    for step_id, text in zip(contract.procedure_step_ids, contract.procedure):
        items.append(SemanticItem(step_id, "procedure", "procedure",
                                  "required" if step_id in contract.required_procedure_step_ids else "recommended",
                                  step_id in contract.required_procedure_step_ids, (), contract.source_trial_ids, text))
    for i, text in enumerate(contract.applicability):
        items.append(SemanticItem(f"scope-{i+1}", "applicability", "scope", "required", True, (), (), text))
    for section, kind, values in (
        ("invariants", "invariant", contract.protected_semantics),
        ("preconditions", "precondition", contract.preconditions),
        ("postconditions", "postcondition", contract.postconditions),
    ):
        for i, text in enumerate(values):
            items.append(SemanticItem(f"{section}-{i+1}", section, kind, "required", True, (), (), text))
    items.append(SemanticItem(
        "legacy-stop-replan", "recovery", "recovery", "required", True, (), (),
        "Stop or replan when a precondition is unknown or a required action is unavailable.",
    ))
    bindings = semantic_atom_bindings or {}
    if set(bindings) - {item.item_id for item in items}:
        raise ValueError("semantic atom binding references an unknown legacy item")
    bound_items = []
    for item in items:
        atoms = tuple(bindings.get(item.item_id, ()))
        if atoms:
            if item.status != "required":
                raise ValueError("semantic atoms may bind only existing required items")
            bound_items.append(replace(item, semantic_atoms=atoms))
        else:
            bound_items.append(item)
    # Construction validates every explicit binding against the vetted exact renderings.
    return SemanticDocument(contract, tuple(bound_items), source_body=contract.body, token_counter=token_counter)


def apply_semantic_delta(document: SemanticDocument, delta: SemanticDelta | Mapping[str, Any]) -> SemanticDocument:
    """Apply one deterministic delta, enforcing semantic, drift, scope and size bounds."""
    if isinstance(delta, Mapping):
        delta = SemanticDelta.from_dict(delta)
    if delta.base_digest != document.digest:
        raise ValueError("semantic delta base digest mismatch")
    if not 1 <= len(delta.operations) <= 3:
        raise ValueError("semantic patch must contain one to three operations")
    if document.refinement_rounds >= 2:
        raise ValueError("semantic refinement limit reached")
    items = list(document.items)
    touched: set[str] = set()
    for operation in delta.operations:
        _validate_operation_text_size(operation)
        by_id = {item.item_id: i for i, item in enumerate(items)}
        if operation.op == "add":
            if operation.item is None or operation.item.item_id in by_id:
                raise ValueError("add requires a new typed item")
            _validate_new_item(operation.item, document.contract)
            if operation.item.status != "recommended" or operation.item.protected or operation.item.section == "invariants":
                raise ValueError("delta additions are optional advice, not new mandatory invariants")
            if operation.after_id is not None and operation.after_id not in by_id:
                raise ValueError("add references an unknown after_id")
            pos = by_id[operation.after_id] + 1 if operation.after_id else len(items)
            items.insert(pos, operation.item)
        elif operation.op == "modify":
            if operation.item_id not in by_id:
                raise ValueError("modify references an unknown item")
            index = by_id[operation.item_id]
            old = items[index]
            if operation.item is None or operation.item.item_id != old.item_id:
                raise ValueError("modify requires a replacement item with the same stable ID")
            candidate = operation.item
            if (candidate.section,candidate.kind,candidate.status,candidate.protected,candidate.semantic_atoms) != (
                old.section,old.kind,old.status,old.protected,old.semantic_atoms
            ):
                raise ValueError("item scope, kind, status, protection and obligation are immutable")
            if old.section in {"applicability", "preconditions", "postconditions"}:
                raise ValueError("scope, preconditions, and postconditions are immutable")
            if old.status == "required" or old.protected:
                if old.status != "required" or not old.semantic_atoms:
                    raise ValueError("required free text has no registered semantic proof")
                if (candidate.section, candidate.kind, candidate.status, candidate.protected, candidate.semantic_atoms) != (
                    old.section, old.kind, old.status, old.protected, old.semantic_atoms
                ):
                    raise ValueError("required obligation, scope, kind, status, and atom order are immutable")
                _validate_required_rendering(candidate)
            _validate_new_item(candidate, document.contract)
            items[index] = candidate
            touched.add(old.item_id)
        elif operation.op == "split":
            if operation.item_id not in by_id or len(operation.new_items) < 2:
                raise ValueError("split requires an existing item and at least two new items")
            index = by_id[operation.item_id]
            old = items[index]
            replacements = operation.new_items
            if old.section in {"applicability", "preconditions", "postconditions"}:
                raise ValueError("scope, preconditions, and postconditions are immutable")
            if replacements[0].item_id != old.item_id or any(x.item_id in by_id and x.item_id != old.item_id for x in replacements[1:]):
                raise ValueError("split must retain the original stable ID on the first part and use new IDs thereafter")
            if old.status == "required" or old.protected:
                if not old.semantic_atoms or any(x.status != "required" for x in replacements):
                    raise ValueError("required free text cannot be split without controlled semantic atoms")
                if any((x.section, x.kind, x.protected) != (old.section, old.kind, old.protected) for x in replacements):
                    raise ValueError("split cannot change required item scope, kind, or protection")
                if tuple(atom for item in replacements for atom in item.semantic_atoms) != old.semantic_atoms:
                    raise ValueError("split must preserve required semantic atoms in their original order")
                for candidate in replacements:
                    _validate_required_rendering(candidate)
            for candidate in replacements:
                _validate_new_item(candidate, document.contract)
            items[index:index+1] = list(replacements)
            touched.add(old.item_id)
        elif operation.op == "merge":
            ids = operation.item_ids
            if len(ids) < 2 or any(item_id not in by_id for item_id in ids) or operation.item is None:
                raise ValueError("merge requires multiple existing items and a merged item")
            positions = [by_id[item_id] for item_id in ids]
            if positions != list(range(min(positions), min(positions) + len(positions))):
                raise ValueError("merge items must be contiguous and supplied in document order")
            originals = [items[pos] for pos in positions]
            merged = operation.item
            if (merged.section,merged.kind) != (originals[0].section,originals[0].kind):
                raise ValueError("merge must preserve section and kind")
            if originals[0].section in {"applicability", "preconditions", "postconditions"}:
                raise ValueError("scope, preconditions, and postconditions are immutable")
            if merged.item_id != originals[0].item_id or any(x.section != originals[0].section for x in originals):
                raise ValueError("merge must preserve first ID and section scope")
            required = any(x.status == "required" for x in originals)
            if any(x.protected and x.status != "required" for x in originals):
                raise ValueError("protected items cannot be merged as optional advice")
            if any((x.kind, x.status) != (originals[0].kind, originals[0].status) for x in originals):
                raise ValueError("merge cannot combine different item kinds or statuses")
            if required:
                if any(x.status != "required" or not x.semantic_atoms for x in originals):
                    raise ValueError("required merge needs atom-backed required items")
                if any((x.kind, x.protected) != (originals[0].kind, originals[0].protected) for x in originals):
                    raise ValueError("merge cannot combine different required kinds or protection")
                atoms = tuple(atom for x in originals for atom in x.semantic_atoms)
                if merged.semantic_atoms != atoms or merged.status != "required" or not merged.protected:
                    raise ValueError("merge must preserve required atoms, order, and protection")
                _validate_required_rendering(merged)
            _validate_new_item(merged, document.contract)
            items[min(positions):max(positions)+1] = [merged]
            touched.update(ids)
        elif operation.op == "delete":
            if operation.item_id not in by_id:
                raise ValueError("delete references an unknown item")
            old = items[by_id[operation.item_id]]
            if old.status == "required" or old.protected:
                raise ValueError("required or protected items cannot be deleted")
            items.remove(old)
            touched.add(old.item_id)
        elif operation.op == "reorder":
            selected = [i for i in items if i.section == operation.section]
            ids = tuple(i.item_id for i in selected)
            if not selected or set(operation.order) != set(ids) or len(operation.order) != len(ids):
                raise ValueError("reorder must name every item in one section exactly once")
            required_ids = [i.item_id for i in selected if i.status == "required"]
            if [i for i in operation.order if i in required_ids] != required_ids:
                raise ValueError("reorder changes required item order")
            ordered = {i.item_id: i for i in selected}
            ordered_items = [ordered[i] for i in operation.order]
            cursor = 0
            for i, item in enumerate(items):
                if item.section == operation.section:
                    items[i] = ordered_items[cursor]
                    cursor += 1
            touched.update(i for old, new in zip(ids, operation.order) if old != new)
        else:
            raise ValueError("unknown semantic operation")

    existing_ids = {item.item_id for item in document.items}
    if len(touched & existing_ids) / max(1, len(existing_ids)) > 0.30:
        raise ValueError("semantic patch changes more than 30 percent of existing items")
    previous = document.to_contract()
    result = replace(document, items=tuple(items), refinement_rounds=document.refinement_rounds + 1,
                     parent_contract_digest=previous.digest,parent_contract_version=previous.version,
                     revision_sources=document.revision_sources + ("semantic_delta",))
    if result.token_count > 2000:
        raise ValueError("rendered semantic document exceeds 2000 counted tokens")
    _validate_document_guards(result)
    return result


def _item_from_dict(value: Any) -> SemanticItem:
    if isinstance(value, SemanticItem):
        return value
    if not isinstance(value, Mapping) or set(value) != {
        "item_id", "section", "kind", "status", "protected", "semantic_tags", "evidence", "text", "semantic_atoms"
    }:
        raise ValueError("typed semantic item fields are incomplete or unexpected")
    tuples = {name: _string_tuple(value[name], name) for name in ("semantic_tags", "evidence", "semantic_atoms")}
    return SemanticItem(value["item_id"], value["section"], value["kind"], value["status"], value["protected"],
                        tuples["semantic_tags"], tuples["evidence"], value["text"], tuples["semantic_atoms"])


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"semantic {name} must be a list of strings")
    return tuple(value)


def _validate_operation_text_size(operation: SemanticOperation) -> None:
    item_texts = [item.text for item in (operation.item, *operation.new_items) if item is not None]
    if sum(map(len, item_texts)) > 250:
        raise ValueError("semantic operation exceeds 250 text characters")


def _validate_required_rendering(item: SemanticItem) -> None:
    # Composite obligations use one exact approved phrase per atom, in atom order.
    variants = {" ".join(parts) for parts in product(*(SEMANTIC_ATOMS[atom] for atom in item.semantic_atoms))}
    if item.text not in variants:
        raise ValueError("required wording is not a vetted rendering of its semantic atoms")


def _validate_new_item(item: SemanticItem, contract: SkillContract) -> None:
    if item.section in {"applicability", "preconditions", "postconditions"}:
        raise ValueError("scope, preconditions, and postconditions cannot be added or rewritten")
    content = item.text.casefold()
    if _INSTANCE.search(item.text):
        raise ValueError("instance-specific entity leakage")
    if any(value and value.casefold() in content for value in contract.source_task_ids + contract.source_trial_ids):
        raise ValueError("source identifier leakage")
    if re.search(r"(?i)(?:\b[A-Z]:\\|/(?:home|Users|mnt|private|tmp)/|api[_ -]?key|password|secret|hidden state|oracle|high_pddl|traj_data)", item.text):
        raise ValueError("private, secret, or oracle leakage")
    if item.status == "required" and item.semantic_atoms:
        _validate_required_rendering(item)


def _validate_document_guards(document: SemanticDocument) -> None:
    if not any(item.section == "procedure" for item in document.items):
        raise ValueError("semantic document must retain a procedure")
    errors = validate_skill_text(document.to_contract(), max_characters=None)
    if errors:
        raise ValueError("semantic candidate rejected: " + ",".join(errors))
