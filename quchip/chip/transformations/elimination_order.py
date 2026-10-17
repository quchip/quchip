"""Default order of a chain of device eliminations (PHYSICS.md §10.8).

Each step of a chain is a second-order reduction, so the order sets which
fourth-order terms the chain keeps and how far the operators of kept devices
spread along the captured maps. :class:`EliminationOrder` chooses each step
from the patch structure of the source chip and the current diagonal
transition energies. It is a cost-and-accuracy heuristic for dispersive chips
and does not bound the error.
"""

from __future__ import annotations

from itertools import combinations
from math import prod
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from quchip.utils.jax_utils import contains_tracer

if TYPE_CHECKING:
    from quchip.chip.chip import Chip

# A spectator waits while a spectator neighbour links it to a kept device that
# is this many times closer to it in transition frequency than the neighbour.
_PRECEDENCE_RATIO = 2.0
# The `g_over_delta < 0.1` validity limit of a fold.
_FOLD_LIMIT = 0.1


def _concrete(value: Any, label: str, caller: str) -> complex:
    """Return a concrete complex value, or raise because the default order cannot compare traced values."""
    if contains_tracer(value):
        raise ValueError(
            f"{caller}: the default elimination order compares transition frequencies and exchange elements, and "
            f"those of {label!r} are traced. Pass order=, for example the eliminated_labels of a call without "
            "tracing."
        )
    return complex(np.asarray(value))


class EliminationOrder:
    """Choose each spectator of a chain of eliminations, and follow the patches of the chain.

    The model starts from the rules of the local patch (PHYSICS.md §10.7) on
    the source chip. Each step links the survivors of its patch, as the
    effective terms of the step do, and moves each kept device's operator
    support onto the patch survivors. When the step reads a level-dependent
    interaction, its correction can depend on levels on the diagonal, and
    the model assumes this dependence (:meth:`remove`). This conservative
    rule can make a patch larger than the local route's, and a larger patch
    can change later choices. The model reads no reduced chip, so
    ``local=True`` and ``local=False`` get the same order.

    Parameters
    ----------
    chip
        The source chip.
    kept
        Labels of the devices that the chain keeps.
    """

    def __init__(self, chip: "Chip", kept: tuple[str, ...]) -> None:
        from quchip.chip.effective import retained_support
        from quchip.chip.transformations.eliminate_device import (
            _coupling_level_dependent,
            _holds_hamiltonian,
            _level_dependent,
        )
        from quchip.engine.assembly import _resolve_system

        labels = tuple(device.label for device in chip.devices)
        self._index = {label: index for index, label in enumerate(labels)}
        self._dims = dict(zip(labels, chip.authored_dims))
        self.kept = frozenset(kept)
        bases = _resolve_system(chip, chip.backend).bases
        # Devices that share a coupling, an effective Hamiltonian or a port, and the
        # devices that a patch adds when its core holds a device.
        self._neighbours: dict[str, set[str]] = {label: set() for label in labels}
        self._reach: dict[str, set[str]] = {label: set() for label in labels}
        for coupling in chip.couplings:
            ends = (coupling.device_a_label, coupling.device_b_label)
            for inner, outer in (ends, ends[::-1]):
                self._neighbours[inner].add(outer)
                if _coupling_level_dependent(chip, coupling, inner, bases):
                    self._reach[inner].add(outer)
        for terms in chip.effective_terms:
            if _holds_hamiltonian(terms):
                for label in terms.labels:
                    self._neighbours[label].update(terms.labels)
                    if _level_dependent(chip, terms.expression(), terms.labels, label, bases):
                        self._reach[label].update(terms.labels)
        # Each port as (targets, devices its operator acts on in retained coordinates).
        self._ports: list[tuple[frozenset[str], frozenset[str]]] = []
        for port in chip.ports:
            targets = port.resolve_targets(chip)
            for label in targets:
                self._neighbours[label].update(targets)
            self._ports.append((frozenset(targets),
                                frozenset(retained_support(chip, targets, f"port:{port.label}"))))
        for label in labels:
            self._neighbours[label].discard(label)
            self._reach[label].discard(label)
        self._supports = {label: frozenset(retained_support(chip, (label,))) for label in kept}
        self.remaining = self.reachable(set(labels) - self.kept)

    def reachable(self, spectators: set[str]) -> set[str]:
        """Return the spectators that share a chain of neighbours with a kept device."""
        found, frontier = set(self.kept), list(self.kept)
        while frontier:
            for neighbour in self._neighbours[frontier.pop()]:
                if neighbour not in found:
                    found.add(neighbour)
                    frontier.append(neighbour)
        return spectators & found

    def patch(self, label: str) -> frozenset[str]:
        """Return the devices that the step of ``label`` reads, as ``eliminate(..., local=True)`` would."""
        core = {label} | self._neighbours[label]
        patch = set(core).union(*(self._reach[member] for member in core))
        for targets, support in self._ports:
            if label in targets:
                patch |= support
        return frozenset(patch)

    def states(self, labels: frozenset[str]) -> int:
        """Return the product dimension of ``labels`` in authored coordinates."""
        return prod(self._dims[label] for label in labels)

    def supports(self, label: str | None = None) -> dict[str, frozenset[str]]:
        """Return the support of each kept device's operator, after the step of ``label`` when it is given."""
        if label is None:
            return dict(self._supports)
        patch = self.patch(label)
        survivors = patch - {label}
        return {kept: (support - patch) | survivors if support & patch else support
                for kept, support in self._supports.items()}

    def choose(self, read: Callable[[tuple[str, ...]], tuple[dict[str, Any], dict[tuple[str, str], Any]]],
               caller: str) -> tuple[str, str | None]:
        """Return the next spectator and, when it has to break a wait, a note that says what the chain omits.

        A spectator ``v`` waits while a spectator neighbour ``u`` has a kept
        neighbour ``k`` with ``|E_v - E_k| < |E_v - E_u| / 2``, so that the
        chain keeps the ``k``-``v`` exchange shift. It also waits while its
        step would give two spectator neighbours ``a`` and ``b`` an exchange
        ``x_ab + x_av x_vb (1/(E_a - E_v) + 1/(E_b - E_v)) / 2`` that is
        nonzero and at least ``0.1 |E'_a - E'_b|``, the fold limit. This
        screens the predicted fold of the pair; it does not make every later
        fold valid. ``E'_a = E_a + |x_av|^2 / (E_a - E_v)`` is the estimated
        transition of ``a`` after the step. ``v`` also waits when ``E_a`` or
        ``E_b`` equals ``E_v``. The pair does not stop the step when ``x_ab``
        alone is nonzero and already reaches the limit. ``read``
        returns the current diagonal transition energies ``E`` and the
        exchange elements ``x`` of the labels it receives, in GHz. Among the
        spectators that do not wait, or all of
        them when every spectator waits, the rule takes a device with a
        spectator neighbour first, then the smallest widest kept-device
        support after the step, the smallest sum of these supports, the
        smallest patch and chip order.
        """
        position = self._index.__getitem__
        links, pairs = {}, {}
        for v in self.remaining:
            spectators = sorted(self._neighbours[v] & self.remaining, key=position)
            links[v] = [(u, k) for u in spectators for k in sorted(self._neighbours[u] & self.kept, key=position)]
            pairs[v] = list(combinations(spectators, 2))
        needed = tuple(sorted({label for v in self.remaining for group in (*links[v], *pairs[v])
                               for label in (v, *group)}, key=position))
        energies: dict[str, float] = {}
        elements: dict[tuple[str, str], complex] = {}
        if needed:
            gaps, raw = read(needed)
            for label, value in gaps.items():
                energies[label] = _concrete(value, label, caller).real
            for (a, b), value in raw.items():
                elements[a, b] = _concrete(value, a, caller)

        def element(a: str, b: str) -> complex:
            if position(a) < position(b):
                return elements.get((a, b), 0.0)
            return elements.get((b, a), 0.0).conjugate()

        def wait(v: str) -> str | None:
            for u, k in links[v]:
                if abs(energies[v] - energies[k]) < abs(energies[v] - energies[u]) / _PRECEDENCE_RATIO:
                    return (f"Eliminated {v!r} while {u!r} still linked it to the kept {k!r}, because every "
                            f"remaining spectator had to wait. The chain omits the fourth-order {k!r}-{v!r} "
                            f"exchange shift through {u!r}.")
            for a, b in pairs[v]:
                direct = element(a, b)
                if direct != 0 and abs(direct) >= _FOLD_LIMIT * abs(energies[a] - energies[b]):
                    continue  # The pair already exceeds the limit, so waiting does not help.
                da, db = energies[a] - energies[v], energies[b] - energies[v]
                if da == 0 or db == 0:
                    return pair_note(v, a, b)
                xa, xb = element(a, v), element(v, b)
                exchange = direct + xa * xb * (1 / da + 1 / db) / 2
                # The step also shifts both transitions, which can bring the pair closer.
                gap = energies[a] + abs(xa) ** 2 / da - energies[b] - abs(xb) ** 2 / db
                if exchange != 0 and abs(exchange) >= _FOLD_LIMIT * abs(gap):
                    return pair_note(v, a, b)
            return None

        def pair_note(v: str, a: str, b: str) -> str:
            return (f"Eliminated {v!r}, which links {a!r} and {b!r}, because every remaining spectator had to "
                    "wait. A later fold of this pair can exceed g/Δ = 0.1; read its validity.")

        waits = {v: wait(v) for v in self.remaining}
        free = [v for v in self.remaining if waits[v] is None] or list(self.remaining)

        def cost(v: str) -> tuple[Any, ...]:
            sizes = [self.states(support) for support in self.supports(v).values()]
            return (self._neighbours[v] <= self.kept, max(sizes, default=1), sum(sizes),
                    self.states(self.patch(v)), position(v))

        target = min(free, key=cost)
        return target, waits[target]

    def remove(self, label: str) -> None:
        """Record the step of ``label``: link its patch survivors and move the supports that meet the patch.

        When a neighbour ``s`` of ``label`` or ``label`` itself has an
        interaction whose diagonal depends on the level of a device ``t``, the
        denominators of the ``s``-``label`` processes read that level. The
        correction on the survivors can then depend on the levels of ``s``
        and ``t`` on its diagonal, so later patches that hold ``s`` or ``t``
        hold every survivor.
        """
        patch = self.patch(label)
        survivors = patch - {label}
        dependent: set[str] = set()
        for member in self._neighbours[label]:
            reached = (self._reach[label] | self._reach[member]) - {label, member}
            if reached:
                dependent |= {member} | reached
        self._supports = self.supports(label)
        self._ports = [(survivors, survivors) if label in targets
                       else (targets, (support - patch) | survivors if support & patch else support)
                       for targets, support in self._ports]
        del self._neighbours[label], self._reach[label]
        for members in (*self._neighbours.values(), *self._reach.values()):
            members.discard(label)
        for member in survivors:
            self._neighbours[member] |= survivors - {member}
        for member in dependent:
            self._reach[member] |= survivors - {member}
        self.remaining.discard(label)
