"""Composable, instantaneous SLH field boundaries."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quchip.chip.ports import Port
from quchip.engine.field_noise import amplifier_values, attenuation_value, noise_parameters, thermal_occupation_value
from quchip.engine.reference import (
    FieldChannel,
    ReferenceAmplifier,
    ReferenceDelay,
    ReferenceElement,
    ReferenceFilter,
    ReferencePlane,
    ReferenceLoss,
    noise_density,
    has_colored_noise,
    source_occupation,
)
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar, select_array_module
from quchip.utils.labeling import auto_label, resolve_label
from quchip.utils.values import copy_value
from quchip.utils.deprecation import warn_renamed

if TYPE_CHECKING:
    from quchip.control.field import CoherentInput
    from quchip.engine.ir import ResolvedSLH
    from quchip.observables import OutputField


TerminalDirection = Literal["input", "output"]
TerminalKey = tuple[str, str]


@dataclass(frozen=True)
class FieldTerminal:
    """One directional terminal owned by a :class:`PortNetwork`.

    Parameters
    ----------
    component : str
        Owning component label.
    name : str
        Component-local terminal name.
    direction : {"input", "output"}
        Field-propagation direction.
    """

    component: str
    name: str
    direction: TerminalDirection
    _network_token: object = field(repr=False, compare=False)

    @property
    def key(self) -> TerminalKey:
        """Return the stable component-local terminal key."""
        return (self.component, self.name)


@dataclass(frozen=True)
class SLHComponent:
    """Boundary component with scalar scattering and named terminals.

    Parameters
    ----------
    label : str
        Unique component label.
    input_names, output_names : tuple[str, ...]
        Terminal names in scattering column and row order.
    scattering : array-like
        Dimensionless scalar scattering matrix.
    sides : tuple[str, ...], default=()
        Names pairing physical input and output terminals.
    """

    label: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    scattering: Any
    _network_token: object = field(repr=False, compare=False)
    sides: tuple[str, ...] = ()
    _local_ports: tuple[Port | None, ...] = field(default=(), repr=False, compare=False)
    _hidden_pairs: tuple[tuple[str, str, str], ...] = field(
        default=(), repr=False, compare=False
    )
    _row_inputs: tuple[tuple[int, ...], ...] | None = field(default=None, repr=False, compare=False)
    _kind: str = field(default="scattering", repr=False)
    _parameters: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _transfer: Callable[..., Any] | None = field(default=None, repr=False, compare=False)

    def port(self, name: str | int) -> "ComponentPort":
        """Return a component port with both field directions.

        Parameters
        ----------
        name : str or int
            Physical side name.
        """
        label = str(name)
        if not self.sides:
            raise ValueError(
                f"Component {self.label!r} is directional and has no component ports; use "
                "input_terminal()/output_terminal() or cascade()."
            )
        if label not in self.sides:
            raise KeyError(f"No port {label!r} on component {self.label!r}; available: {list(self.sides)}")
        return ComponentPort(self.input_terminal(label), self.output_terminal(label))

    def side(self, name: str | int) -> "ComponentPort":
        """Deprecated alias for :meth:`port`.

        Parameters
        ----------
        name : str or int
            Physical side name.
        """
        warn_renamed("component.side()", "component.port()")
        return self.port(name)

    @property
    def inputs(self) -> tuple[FieldTerminal, ...]:
        """Return input terminals in the component's scattering-column order."""
        return tuple(
            FieldTerminal(self.label, name, "input", self._network_token)
            for name in self.input_names
        )

    @property
    def outputs(self) -> tuple[FieldTerminal, ...]:
        """Return output terminals in the component's scattering-row order."""
        return tuple(
            FieldTerminal(self.label, name, "output", self._network_token)
            for name in self.output_names
        )

    def input_terminal(self, name: str) -> FieldTerminal:
        """Return one named input terminal.

        Parameters
        ----------
        name : str
            Input terminal name.
        """
        try:
            index = self.input_names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"No input {name!r} on component {self.label!r}; available: {list(self.input_names)}"
            ) from exc
        return self.inputs[index]

    def output_terminal(self, name: str) -> FieldTerminal:
        """Return one named output terminal.

        Parameters
        ----------
        name : str
            Output terminal name.
        """
        try:
            index = self.output_names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"No output {name!r} on component {self.label!r}; available: {list(self.output_names)}"
            ) from exc
        return self.outputs[index]

    @property
    def input(self) -> FieldTerminal:
        """Return the sole or signal input terminal."""
        if "signal" in self.input_names:
            return self.input_terminal("signal")
        if len(self.input_names) == 1:
            return self.inputs[0]
        raise AttributeError(
            f"Component {self.label!r} has multiple inputs; use input_terminal(name)."
        )

    @property
    def output(self) -> FieldTerminal:
        """Return the sole or signal output terminal."""
        if "signal" in self.output_names:
            return self.output_terminal("signal")
        if len(self.output_names) == 1:
            return self.outputs[0]
        raise AttributeError(
            f"Component {self.label!r} has multiple outputs; use output_terminal(name)."
        )


@dataclass(frozen=True)
class ComponentPort:
    """A component port pairing incoming and outgoing field connections.

    Parameters
    ----------
    input, output : FieldTerminal
        Incoming and outgoing terminals on the same physical side.
    """

    input: FieldTerminal
    output: FieldTerminal


@dataclass(frozen=True, eq=False)
class NetworkPort:
    """A named external network port with incoming and outgoing fields.

    The reference plane specifies where these fields are defined. A device's
    :class:`Port` instead specifies its coupling to the network.

    Parameters
    ----------
    label : str
        External reference-plane label.
    """

    label: str
    _input_key: TerminalKey = field(repr=False)
    _output_key: TerminalKey = field(repr=False)
    _hidden: bool = field(default=False, repr=False)
    _network_token: object = field(repr=False, compare=False, kw_only=True)

    def __eq__(self, other: object) -> bool:
        """Compare stable plane identity within one network."""
        return (
            isinstance(other, NetworkPort)
            and self._network_token is other._network_token
            and self.label == other.label
        )

    def __hash__(self) -> int:
        """Hash stable plane identity within one network."""
        return hash((id(self._network_token), self.label))

    @property
    def input(self) -> "CoherentInput":
        """Return the coherent-field endpoint scheduled at this plane."""
        if self._hidden:
            raise AttributeError("Hidden vacuum channels cannot be driven.")
        from quchip.control.field import CoherentInput

        return CoherentInput(self.label, label=f"{self.label}.input")

    @property
    def output(self) -> "OutputField":
        """Return the complete transient output-field request for this plane."""
        if self._hidden:
            raise AttributeError("Hidden vacuum channels are not observable.")
        from quchip.observables import OutputField

        return OutputField(self.label)


def _strongly_connected(nodes: Mapping[Any, Any], successors: Callable[[Any], list[Any]]) -> list[list[Any]]:
    """Return strongly connected components with every component after the ones it depends on."""
    index: dict[Any, int] = {}
    lowlink: dict[Any, int] = {}
    stack: list[Any] = []
    on_stack: set[Any] = set()
    components: list[list[Any]] = []

    def visit(node: Any) -> None:
        index[node] = lowlink[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        for successor in successors(node):
            if successor not in index:
                visit(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in on_stack:
                lowlink[node] = min(lowlink[node], index[successor])
        if lowlink[node] == index[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            components.append(component[::-1])

    for node in nodes:
        if node not in index:
            visit(node)
    return components


@dataclass(frozen=True)
class IncludedNetwork:
    """Access one prefixed copy of a template network inside a host.

    ``component(name)`` returns a copied component. Template exposures made with
    ``expose(..., at=...)`` are available through ``side(name)``; asymmetric
    exposures use ``input(name)`` and ``output(name)``.

    Parameters
    ----------
    prefix : str
        Label prefix assigned to the included copy.
    """

    prefix: str
    _host: "PortNetwork" = field(repr=False, compare=False)
    _interfaces: Mapping[str, tuple[TerminalKey, TerminalKey]] = field(repr=False, compare=False)

    def component(self, name: str) -> SLHComponent:
        """Return a copied template component.

        Parameters
        ----------
        name : str
            Unprefixed template component name.
        """
        label = f"{self.prefix}/{name}"
        if label not in self._host._components:
            raise KeyError(f"No component {name!r} in block {self.prefix!r}.")
        return self._host._components[label]

    def input(self, name: str) -> FieldTerminal:
        """Return an exported input terminal.

        Parameters
        ----------
        name : str
            Template external-port name.
        """
        key = self._interface(name)[0]
        return self._host._components[key[0]].input_terminal(key[1])

    def output(self, name: str) -> FieldTerminal:
        """Return an exported output terminal.

        Parameters
        ----------
        name : str
            Template external-port name.
        """
        key = self._interface(name)[1]
        return self._host._components[key[0]].output_terminal(key[1])

    def port(self, name: str) -> ComponentPort:
        """Return an exported component port.

        Parameters
        ----------
        name : str
            Symmetric template external-port name.
        """
        input_key, output_key = self._interface(name)
        component = self._host._components[input_key[0]]
        if input_key != output_key or input_key[1] not in component.sides:
            raise ValueError(f"Interface {name!r} of block {self.prefix!r} is asymmetric; use input() and output().")
        return component.port(input_key[1])

    def side(self, name: str) -> ComponentPort:
        """Deprecated alias for :meth:`port`.

        Parameters
        ----------
        name : str
            Symmetric template external-port name.
        """
        warn_renamed("block.side()", "block.port()")
        return self.port(name)

    def _interface(self, name: str) -> tuple[TerminalKey, TerminalKey]:
        if name not in self._interfaces:
            raise KeyError(f"No interface {name!r} in block {self.prefix!r}; available: {list(self._interfaces)}")
        return self._interfaces[name]


_SERIALIZED_FACTORIES = frozenset(
    {
        "through",
        "hybrid90",
        "permutation",
        "circulator",
        "isolator",
        "phase_shift",
        "beam_splitter",
        "attenuator",
        "delay",
        "amplifier",
        "termination",
    }
)


#: ``(coefficient, boundary_input, upstream_output)`` for one structurally fed column.
_Feeder = tuple[Any, "TerminalKey | None", "TerminalKey | None"]


@dataclass
class _AffineField:
    scattering: list[Any]
    coupling: dict[str, Any]
    support: list[bool]


@dataclass(frozen=True)
class _CompiledChannel:
    exposure: NetworkPort
    coupling: Mapping[str, Any]
    reference: ReferencePlane


@dataclass(frozen=True)
class _CompiledNetwork:
    channels: tuple[_CompiledChannel, ...]
    scattering: Any
    generated_pairs: tuple[tuple[str, str, Any], ...]
    support: np.ndarray
    output_network: Any = None


class PortNetwork:
    """Compose Markovian ports with instantaneous scalar scattering.

    The network acts on fields at one reference frequency. Scattering matrices
    are dimensionless and must be unitary; loss and noise are represented by
    explicit hidden channels or reference components. Feedback loops are
    reduced algebraically, so this class does not model propagation retardation.

    Parameters
    ----------
    scattering : array-like or mapping, optional
        Authored instantaneous scattering matrix or labeled entries.
    label : str, optional
        Network label.
    """

    _type_prefix = "port_network"

    def __init__(
        self,
        *,
        scattering: Any = None,
        label: str | None = None,
    ) -> None:
        """Create an empty network, optionally with an authored scattering map.

        Parameters
        ----------
        scattering : array-like or mapping, optional
            Static scattering matrix, or ``{(output, input): amplitude}``
            entries keyed by external labels. A callable is rejected because
            the network is instantaneous.
        label : str, optional
            Network label; generated when omitted.
        """
        self.label = label if label is not None else auto_label(self._type_prefix)
        self._token = object()
        if callable(scattering):
            raise TypeError("PortNetwork scattering is instantaneous; callables are not supported.")
        if isinstance(scattering, Mapping):
            normalized: dict[tuple[str, str], Any] = {}
            for key, value in scattering.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise TypeError("Scattering mappings use (output, input) keys.")
                output, input_ = (resolve_label(item) for item in key)
                if (output, input_) in normalized:
                    raise ValueError(f"Duplicate scattering entry {(output, input_)}.")
                normalized[(output, input_)] = value
            self._authored_scattering = normalized
        else:
            self._authored_scattering = scattering
        self._components: dict[str, SLHComponent] = {}
        self._ports: list[Port] = []
        self._connections: dict[TerminalKey, TerminalKey] = {}
        self._used_outputs: dict[TerminalKey, TerminalKey] = {}
        self._exposures: list[NetworkPort] = []

    @classmethod
    def from_ports(
        cls,
        ports: Sequence[Port],
        *,
        scattering: Any = None,
        label: str | None = None,
    ) -> "PortNetwork":
        """Build a network containing ``ports`` in declaration order.

        Parameters
        ----------
        ports : sequence of Port
            Quantum coupling channels to add. Each port may belong to only one
            network; pass a copy to reuse it elsewhere.
        scattering : array-like or mapping, optional
            Initial external scattering specification.
        label : str, optional
            Network label.

        Returns
        -------
        PortNetwork
            Network with the supplied ports and no internal components.
        """
        network = cls(scattering=scattering, label=label)
        for port in ports:
            network._add_port(port)
        return network

    @property
    def ports(self) -> tuple[Port, ...]:
        """Return network-owned quantum coupling ports in declaration order."""
        return tuple(self._ports)

    @property
    def components(self) -> tuple[SLHComponent, ...]:
        """Return all graph components in declaration order."""
        return tuple(self._components.values())

    @property
    def external_ports(self) -> tuple[NetworkPort, ...]:
        """Return the external network ports in channel order."""
        return tuple(exposure for exposure in self._effective_exposures() if not exposure._hidden)

    @property
    def exposures(self) -> tuple[NetworkPort, ...]:
        """Deprecated alias for :attr:`external_ports`."""
        warn_renamed("network.exposures", "network.external_ports")
        return self.external_ports

    def exposure(self, exposure: str | NetworkPort) -> NetworkPort:
        """Deprecated alias for :meth:`external_port`.

        Parameters
        ----------
        exposure : str or NetworkPort
            External reference plane.
        """
        warn_renamed("network.exposure()", "network.external_port()")
        return self.external_port(exposure)

    def external_port(self, exposure: str | NetworkPort) -> NetworkPort:
        """Return one external network port by object or label.

        Parameters
        ----------
        exposure : str or NetworkPort
            External reference plane.
        """
        if isinstance(exposure, NetworkPort) and exposure._network_token is not self._token:
            raise ValueError(f"Network port {exposure.label!r} belongs to another PortNetwork.")
        label = resolve_label(exposure)
        exposures = self.external_ports
        for candidate in exposures:
            if candidate.label == label:
                return candidate
        raise KeyError(
            f"No exposure labeled {label!r}. Available: "
            f"{[candidate.label for candidate in exposures]}"
        )

    @property
    def S(self) -> Any:
        """Return the composed instantaneous scalar scattering matrix."""
        return self._compile().scattering

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return bindable scalar network parameters visible in authored scattering."""
        values: dict[str, Any] = {}
        if isinstance(self._authored_scattering, Mapping):
            for (output, input_), value in self._authored_scattering.items():
                values[f"scattering.{resolve_label(output)}.{resolve_label(input_)}"] = value
        for component in self.components:
            for name, value in component._parameters.items():
                values[f"component.{component.label}.{name}"] = value
        return MappingProxyType(values)

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Set one network-owned scalar on an isolated structural copy.

        Parameters
        ----------
        name : str
            Path below ``scattering`` or ``component``.
        value : Any
            Replacement scalar in the component parameter's declared units.
        """
        parts = name.split(".")
        if (
            len(parts) == 3
            and parts[0] == "scattering"
            and isinstance(self._authored_scattering, dict)
        ):
            key = (parts[1], parts[2])
            if key in self._authored_scattering:
                self._authored_scattering[key] = value
                return
        if len(parts) == 3 and parts[0] == "component":
            component = self._components.get(parts[1])
            if component is not None and parts[2] in component._parameters:
                component._parameters[parts[2]] = value
                return
        raise KeyError(name)

    def port(
        self,
        label: str,
        *,
        target: Any | Sequence[Any],
        rate: Any = None,
        external_quality_factor: Any = None,
        operator: Any = None,
        phase: Any = 0.0,
    ) -> Port:
        """Create and add one quantum coupling port.

        Parameters
        ----------
        label : str
            Unique port label.
        target, rate, external_quality_factor, operator, phase
            See :class:`~quchip.chip.ports.Port`.

        Returns
        -------
        Port
            The network-owned port, whose ``input``, ``output`` and ``side``
            terminals can be connected immediately.
        """
        port = Port(
            target,
            rate=rate,
            external_quality_factor=external_quality_factor,
            operator=operator,
            phase=phase,
            label=label,
        )
        self._add_port(port)
        return port

    def component(
        self,
        label: str,
        *,
        scattering: Any,
        terminals: Sequence[str] | None = None,
    ) -> SLHComponent:
        """Add a scalar scattering component with named terminals.

        Parameters
        ----------
        label : str
            Unique component label.
        scattering : square array-like
            Dimensionless scattering matrix. Rows are outputs and columns are
            inputs; concrete matrices must be unitary.
        terminals : sequence of str, optional
            Names for both input and output terminals. Defaults to ``("signal",)``
            for one channel and numeric names otherwise.

        Returns
        -------
        SLHComponent
            Component handle used with :meth:`connect`, :meth:`cascade`, or
            :meth:`link`.
        """
        return self._scattering_component(label, scattering=scattering, terminals=terminals, kind="scattering")

    def _scattering_component(
        self, label: str, *, scattering: Any, terminals: Sequence[str] | None,
        kind: str, parameters: dict[str, Any] | None = None,
    ) -> SLHComponent:
        shape = getattr(scattering, "shape", None)
        if shape is None:
            shape = np.shape(scattering)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"SLHComponent scattering must be square, got shape {shape}.")
        size = int(shape[0])
        names = tuple(terminals) if terminals is not None else (
            ("signal",) if size == 1 else tuple(str(index) for index in range(size))
        )
        if len(names) != size or len(set(names)) != len(names):
            raise ValueError("Component terminal names must be unique and match scattering size.")
        component = SLHComponent(
            label=label,
            input_names=names,
            output_names=names,
            scattering=scattering,
            _network_token=self._token,
            _local_ports=tuple(None for _ in names),
            _kind=kind,
            _parameters={} if parameters is None else parameters,
        )
        return self._add_component(component)

    def through(self, label: str) -> SLHComponent:
        """Add a one-channel identity through component.

        Parameters
        ----------
        label : str
            Component label.
        """
        return self._scattering_component(label, scattering=[[1.0]], terminals=("signal",), kind="through")

    def phase_shift(self, label: str, *, phase: Any) -> SLHComponent:
        """Add a one-channel phase shift.

        Parameters
        ----------
        label : str
            Component label.
        phase : float or array-like
            Phase in radians.
        """
        xp = select_array_module(contains_tracer(phase))
        return self._scattering_component(
            label,
            scattering=xp.asarray([[xp.exp(1j * xp.asarray(phase))]]),
            terminals=("signal",),
            kind="phase_shift",
            parameters={"phase": phase},
        )

    def beam_splitter(self, label: str, *, eta: Any = 0.5) -> SLHComponent:
        """Add a directional two-input/two-output splitter.

        ``eta`` is the power from input ``k`` to output ``k``. With
        ``t = sqrt(eta)`` and ``r = sqrt(1 - eta)``, the scattering matrix is
        ``[[t, r], [-r, t]]``. This component has no component ports; use
        ``input_terminal()``, ``output_terminal()``, or ``cascade()``.

        Parameters
        ----------
        label : str
            Component label.
        eta : float or array-like, default=0.5
            Power transmission in [0, 1].
        """
        matrix = self._transmission_matrix(eta)
        return self._scattering_component(
            label, scattering=matrix, terminals=("left", "right"), kind="beam_splitter", parameters={"eta": eta}
        )

    def hybrid90(self, label: str) -> SLHComponent:
        """Add a directional two-input/two-output ideal 90-degree hybrid.

        The scattering matrix is ``[[1, 1j], [1j, 1]] / sqrt(2)``. This
        component has no component ports; use ``input_terminal()``,
        ``output_terminal()``, or ``cascade()``.

        Parameters
        ----------
        label : str
            Unique component label.
        """
        return self._scattering_component(
            label,
            scattering=np.asarray([[1.0, 1j], [1j, 1.0]], dtype=complex) / np.sqrt(2.0),
            terminals=("left", "right"),
            kind="hybrid90",
        )

    def permutation(self, label: str, *, order: Sequence[int]) -> SLHComponent:
        """Add an output permutation.

        Parameters
        ----------
        label : str
            Unique component label.
        order : sequence of int
            Input column selected by each output row.
        """
        order = tuple(order)
        if sorted(order) != list(range(len(order))):
            raise ValueError("Permutation order must contain each input index exactly once.")
        names = tuple(str(index) for index in range(len(order)))
        return self._permutation_component(label, names, order, kind="permutation", sided=False)

    def attenuator(
        self, label: str, *, eta: Any = None, loss_db: Any = None, thermal_occupation: Any = None,
    ) -> SLHComponent:
        """Add a reciprocal two-sided attenuator with power transmission ``eta``.

        Each direction has amplitude transmission ``sqrt(eta)`` and couples to
        one of two hidden loss channels with amplitude ``sqrt(1-eta)``.
        They default to vacuum. ``thermal_occupation`` gives both loads a mean
        thermal population in quanta, constant across the modeled band.
        Specify either power transmission ``eta`` or positive ``loss_db``.

        Parameters
        ----------
        label : str
            Unique component label.
        eta : scalar or None, default=None
            Power transmission in the interval [0, 1].
        loss_db : scalar or None, default=None
            Positive power loss in dB, mutually exclusive with ``eta``.
        thermal_occupation : scalar or None, default=None
            Mean occupation of each hidden load in quanta; ``None`` means vacuum.
        """
        power = {key: value for key, value in (("eta", eta), ("loss_db", loss_db)) if value is not None}
        transmission = attenuation_value(power)
        component = SLHComponent(
            label=label,
            input_names=("1", "2", "vacuum_1", "vacuum_2"),
            output_names=("1", "2", "vacuum_1", "vacuum_2"),
            scattering=self._attenuator_matrix(transmission),
            _network_token=self._token,
            sides=("1", "2"),
            _local_ports=(None,) * 4,
            _hidden_pairs=tuple(
                (name, name, f"hidden.{label}.{name}") for name in ("vacuum_1", "vacuum_2")
            ),
            _row_inputs=((1, 3), (0, 2), (0, 2), (1, 3)),
            _kind="attenuator",
            _parameters={**power, **noise_parameters(thermal_occupation)},
        )
        return self._add_component(component)

    def delay(self, label: str, *, duration: Any) -> SLHComponent:
        """Add a two-sided reference section with duration ``duration`` ns.

        The section shifts fields at exposure planes but does not introduce
        retardation or memory into the Markovian dynamics.

        Place it with :meth:`link` or :meth:`connect` like any other component.
        The compiler peels adjacent runs from each exposure leg, so the section
        never enters Markovian ``S``, ``L``, or ``H``. Every reference section must
        belong to an exposed run or an acyclic downstream output graph. Its duration is tracked at
        ``network.component.<label>.duration``.

        Parameters
        ----------
        label : str
            Component label.
        duration : float or array-like
            Reference-plane delay in ns.
        """
        ReferenceDelay(label, duration)
        return self._reference_component(label, kind="delay", parameters={"duration": duration})

    def filter(
        self, label: str, *, transfer: Callable[..., Any], thermal_occupation: Any = None,
        **parameters: Any,
    ) -> SLHComponent:
        """Add a two-sided passive filter reference section.

        ``transfer(frequency, **parameters)`` must accept scalar or array frequencies
        in GHz and return a broadcast-compatible complex value without concretizing
        traced inputs. Each keyword parameter is tracked at
        ``network.component.<label>.<name>``.

        Place the section with :meth:`link` or :meth:`connect`. The compiler peels it
        from an adjacent exposure leg, so it never enters the Markovian ``S``, ``L``,
        or ``H``: it reshapes the propagating field and leaves every collapse
        operator, including the radiative (Purcell) decay of coupled devices,
        unchanged. Model a Purcell filter as a :class:`~quchip.Resonator` between
        the readout mode and the port instead (see the Purcell filtering guide).
        Continuous-wave APIs evaluate the transfer at each frequency;
        transient APIs use its narrowband value at the relevant carrier. Concrete
        evaluations with ``|H| > 1`` raise. Networks containing filters cannot be
        serialized with :meth:`to_dict`; ``Chip.clone()`` and ``Chip.with_params()``
        preserve the callable.

        Declare ``thermal_occupation`` in quanta, constant across the modeled
        band, for a matched absorptive realization emitting (1-|H|²)n.
        A scalar transfer alone does not distinguish absorption from reflection.
        Colored thermal emission cannot feed a quantum coupling through a
        reference section; use a dynamical filter/bath model for that case.

        Parameters
        ----------
        label : str
            Unique component label.
        transfer : callable
            Passive complex amplitude transfer versus frequency in GHz.
        thermal_occupation : scalar or None, default=None
            Mean matched-load occupation in quanta; ``None`` means vacuum.
        **parameters : Any
            Named transfer-function parameters tracked by the network.
        """
        parameters.update(noise_parameters(thermal_occupation))
        return self._reference_component(label, kind="filter", parameters=dict(parameters), transfer=transfer)

    def amplifier(
        self, label: str, *, added_noise: Any, gain: Any = None, gain_db: Any = None,
    ) -> SLHComponent:
        """Add a phase-preserving amplifier reference section to an output line.

        ``gain`` is power gain ``G``. ``added_noise`` is input-referred
        symmetrized noise in quanta and must be at least ``(1 - 1/G) / 2``.
        Forward propagation from side 1 to side 2 multiplies field amplitudes by
        ``sqrt(G)``; reverse propagation is transparent.

        Place side 1 toward the chip and side 2 toward the exposed output plane.
        The compiler rejects a section that would amplify an incident field into
        the chip or put the output plane on side 1. Both parameters are tracked at
        ``network.component.<label>.<name>`` and remain sweepable and
        differentiable. Acyclic downstream splitters retain its shared output
        noise. The section remains outside Markovian ``S``, ``L``, and
        ``H`` and serializes normally.

        ``gain_db`` may replace linear ``gain``. Added quanta are constant
        across the modeled band and exclude the input's own noise.

        Parameters
        ----------
        label : str
            Component label.
        added_noise : float or array-like
            Input-referred symmetrized added noise in quanta.
        gain, gain_db : float or array-like, optional
            Power gain, specified linearly or in dB; supply at most one.
        """
        parameters = {key: value for key, value in (
            ("gain", gain), ("gain_db", gain_db), ("added_noise", added_noise)) if value is not None}
        ReferenceAmplifier(label, *amplifier_values(parameters))
        return self._reference_component(
            label, kind="amplifier", parameters=parameters,
        )

    def _reference_component(
        self, label: str, *, kind: str, parameters: dict[str, Any], transfer: Callable[..., Any] | None = None,
    ) -> SLHComponent:
        component = SLHComponent(
            label=label,
            input_names=("1", "2"),
            output_names=("1", "2"),
            scattering=None,
            _network_token=self._token,
            sides=("1", "2"),
            _local_ports=(None, None),
            _kind=kind,
            _parameters=parameters,
            _transfer=transfer,
        )
        return self._add_component(component)

    def circulator(self, label: str, *, ports: int = 3) -> SLHComponent:
        """Add an ideal circulator routing side ``k`` to side ``k + 1``.

        The highest-numbered side routes back to side 1.

        Parameters
        ----------
        label : str
            Unique component label.
        ports : int, default=3
            Number of physical sides; must be at least three.
        """
        if ports < 3:
            raise ValueError(f"A circulator needs at least three sides, got {ports}.")
        names = tuple(str(index) for index in range(1, ports + 1))
        sources = tuple((row - 1) % ports for row in range(ports))
        return self._permutation_component(label, names, sources, kind="circulator")

    def isolator(
        self, label: str, *, thermal_occupation: Any = None,
    ) -> SLHComponent:
        """Add an ideal isolator routing side 1 to side 2.

        The reverse field is dumped into ``hidden.<label>.load``. Its thermal
        population travels back toward side 1. ``thermal_occupation`` is in
        quanta, constant across the modeled band; the default is vacuum.

        Parameters
        ----------
        label : str
            Unique component label.
        thermal_occupation : scalar or None, default=None
            Mean hidden-load occupation in quanta; ``None`` means vacuum.
        """
        component = self._permutation_component(
            label,
            ("1", "2", "load"),
            (2, 0, 1),
            kind="isolator",
            hidden_pairs=(("load", "load", f"hidden.{label}.load"),),
        )
        component._parameters.update(noise_parameters(thermal_occupation))
        return component

    def termination(
        self, label: str, *, thermal_occupation: Any = None,
    ) -> SLHComponent:
        """Add a matched one-sided load with an optional thermal input state.

        ``thermal_occupation`` is the mean thermal population in quanta,
        constant across the modeled band. The default is vacuum.

        Parameters
        ----------
        label : str
            Unique component label.
        thermal_occupation : scalar or None, default=None
            Mean load occupation in quanta; ``None`` means vacuum.
        """
        component = self._permutation_component(
            label, ("1", "load"), (1, 0), kind="termination",
            hidden_pairs=(("load", "load", f"hidden.{label}.load"),),
        )
        component._parameters.update(noise_parameters(thermal_occupation))
        return component

    def _permutation_component(
        self,
        label: str,
        names: tuple[str, ...],
        sources: Sequence[int],
        *,
        kind: str,
        hidden_pairs: tuple[tuple[str, str, str], ...] = (),
        sided: bool = True,
    ) -> SLHComponent:
        """Add a selection component whose output row ``r`` copies input ``sources[r]``."""
        size = len(names)
        matrix = np.zeros((size, size), dtype=complex)
        matrix[np.arange(size), sources] = 1.0
        hidden = {pair[0] for pair in hidden_pairs}
        component = SLHComponent(
            label=label,
            input_names=names,
            output_names=names,
            scattering=matrix,
            _network_token=self._token,
            sides=tuple(name for name in names if name not in hidden) if sided else (),
            _local_ports=(None,) * size,
            _hidden_pairs=hidden_pairs,
            _row_inputs=tuple((source,) for source in sources),
            _kind=kind,
        )
        return self._add_component(component)

    def connect(self, output: FieldTerminal, input: FieldTerminal) -> None:
        """Connect one component output to one component input.

        Parameters
        ----------
        output, input : FieldTerminal
            Unused directional terminals from this network.
        """
        self._validate_terminal(output, "output")
        self._validate_terminal(input, "input")
        self._reject_hidden_terminal(output)
        self._reject_hidden_terminal(input)
        if input.key in self._connections:
            raise ValueError(f"Input terminal {input.key} is already connected.")
        if output.key in self._used_outputs:
            raise ValueError(f"Output terminal {output.key} is already connected.")
        if self._terminal_is_exposed(input) or self._terminal_is_exposed(output):
            raise ValueError("An exposed terminal cannot also be connected internally.")
        self._connections[input.key] = output.key
        self._used_outputs[output.key] = input.key

    def link(self, *items: Port | SLHComponent | ComponentPort) -> None:
        """Link consecutive component ports in both directions.

        When a two-sided component is passed directly, the chain enters side 1
        and leaves side 2. Pass ``component.port(k)`` to select a side
        explicitly.

        Parameters
        ----------
        *items : Port, SLHComponent, or ComponentPort
            At least two two-sided endpoints.
        """
        for first, second in self._pairs("link", items):
            left = self._side_of(first, leaving=True)
            right = self._side_of(second, leaving=False)
            self.connect(left.output, right.input)
            self.connect(right.output, left.input)

    def cascade(self, *items: Port | SLHComponent | FieldTerminal) -> None:
        """Connect each item's output to the next item's input in sequence.

        Parameters
        ----------
        *items : Port, SLHComponent, or FieldTerminal
            At least two compatible endpoints.
        """
        for first, second in self._pairs("cascade", items):
            self.connect(self._endpoint_of(first, "output"), self._endpoint_of(second, "input"))

    @staticmethod
    def _pairs(name: str, items: tuple[Any, ...]) -> Iterator[tuple[Any, Any]]:
        if len(items) < 2:
            raise ValueError(f"{name}() requires at least two items, got {len(items)}")
        return pairwise(items)

    def expose(
        self,
        label: str,
        *,
        at: Port | SLHComponent | ComponentPort | None = None,
        input: FieldTerminal | Port | SLHComponent | None = None,
        output: FieldTerminal | Port | SLHComponent | None = None,
    ) -> NetworkPort:
        """Name and return an external network port.

        Pass ``at=`` to expose one component port. Use ``input=`` and
        ``output=`` for separate input and output connections; ports and components then select
        their sole or ``signal`` terminal unless explicit terminals are passed.

        Parameters
        ----------
        label : str
            Unique external reference-plane label.
        at : Port, SLHComponent, ComponentPort, or None, default=None
            Physical side to expose for both directions.
        input, output : FieldTerminal, Port, SLHComponent, or None, default=None
            Separate incoming and outgoing connection points.
        """
        if at is not None:
            if input is not None or output is not None:
                raise TypeError("expose() takes either at= or input=/output=, not both.")
            side = self._side_of(at, leaving=None)
            input, output = side.input, side.output
        if input is None or output is None:
            raise TypeError("expose() requires at= or both input= and output=.")
        input = self._endpoint_of(input, "input")
        output = self._endpoint_of(output, "output")
        self._validate_terminal(input, "input")
        self._validate_terminal(output, "output")
        self._reject_hidden_terminal(input)
        self._reject_hidden_terminal(output)
        if label in {exposure.label for exposure in self._exposures}:
            raise ValueError(f"Duplicate external network port label {label!r}.")
        if input.key in self._connections or output.key in self._used_outputs:
            raise ValueError("Connected terminals cannot also be exposed.")
        if self._terminal_is_exposed(input) or self._terminal_is_exposed(output):
            raise ValueError("A terminal cannot belong to more than one external network port.")
        exposure = NetworkPort(label, input.key, output.key, _network_token=self._token)
        self._exposures.append(exposure)
        return exposure

    def validate_for(self, chip: Any) -> None:
        """Validate every quantum port target against a chip.

        Parameters
        ----------
        chip : Chip
            Chip whose device labels must contain every port target.
        """
        for port in self._ports:
            port.resolve_targets(chip)

    def fingerprint(self) -> tuple[Any, ...]:
        """Return a conservative structural cache signature."""
        return (
            self.label,
            tuple(
                (
                    component.label,
                    component.input_names,
                    component.output_names,
                    component._kind,
                    self._cache_value(component.scattering),
                    tuple(
                        (name, self._cache_value(value))
                        for name, value in component._parameters.items()
                    ),
                    id(component._transfer),
                )
                for component in self.components
            ),
            tuple(self._connections.items()),
            tuple((item.label, item._input_key, item._output_key) for item in self._exposures),
            self._cache_value(self._authored_scattering),
        )

    def resolve(
        self, base: "ResolvedSLH", *, _compiled: _CompiledNetwork | None = None,
    ) -> "ResolvedSLH":
        """Compose this boundary onto an input-free SLH value.

        Parameters
        ----------
        base : ResolvedSLH
            Resolved quantum-port channels to compose with the network.
        """
        from quchip.engine.ir import (
            CollapseTerm,
            HamiltonianProgram,
            ResolvedSLH,
            SLHChannel,
            StaticTerm,
        )

        if not self._ports:
            raise ValueError("PortNetwork requires at least one quantum coupling port.")
        port_channels = {channel.collapse.label: channel for channel in base.external_channels}
        expected = [port.label for port in self._ports]
        if set(port_channels) != set(expected) or len(port_channels) != len(expected):
            raise RuntimeError(
                "PortNetwork channels do not match assembled chip ports; "
                f"network={expected}, assembled={list(port_channels)}."
            )

        compiled = self._compile() if _compiled is None else _compiled
        operators = {label: port_channels[label].coupling for label in expected}
        channels: list[SLHChannel] = []
        for entry, field_channel in zip(compiled.channels, self._field_channels(compiled), strict=True):
            exposure, mapping = entry.exposure, entry.coupling
            input_occupation = field_channel.input_occupation
            frame_frequency = self._common_frame_frequency(
                mapping, port_channels, boundary=f"exposure {exposure.label!r}"
            )
            coupling = self._materialize_coupling(mapping, operators, key=exposure.label)
            if (
                exposure.label in port_channels
                and set(mapping) == {exposure.label}
                and self._is_concrete_one(mapping[exposure.label])
            ):
                channels.append(
                    replace(
                        port_channels[exposure.label],
                        key=exposure.label,
                        accessibility="hidden" if exposure._hidden else "exposed",
                        reference=entry.reference,
                        input_occupation=input_occupation,
                    )
                )
                continue
            template = CollapseTerm(
                operator=coupling,
                rate=1.0,
                source=exposure.label,
                channel="network",
                frame_frequency=frame_frequency,
            )
            channels.append(
                SLHChannel(
                    key=exposure.label,
                    accessibility="hidden" if exposure._hidden else "exposed",
                    collapse=template,
                    coupling_operator=coupling,
                    reference=entry.reference,
                    input_occupation=input_occupation,
                )
            )

        generated = self._materialize_generated_hamiltonian(compiled.generated_pairs, operators)
        static_terms = base.H.static_terms
        if generated is not None:
            static_terms = (*static_terms, StaticTerm(operator=generated, origin="network"))

        hidden = base.hidden_channels
        full_size = len(channels) + len(hidden)
        xp = select_array_module(contains_tracer(compiled.scattering))
        full_support = np.eye(full_size, dtype=bool)
        full_support[: len(channels), : len(channels)] = compiled.support
        full_scattering = xp.eye(full_size, dtype=complex)
        if channels:
            if hasattr(full_scattering, "at"):
                full_scattering = full_scattering.at[: len(channels), : len(channels)].set(compiled.scattering)
            else:
                full_scattering[: len(channels), : len(channels)] = compiled.scattering
        return ResolvedSLH(
            scattering=full_scattering,
            hamiltonian=HamiltonianProgram(
                static_terms=static_terms,
                dynamic_terms=base.H.dynamic_terms,
            ),
            channels=tuple((*channels, *hidden)),
            support=full_support,
            output_network=compiled.output_network,
        )

    def _field_channels(self, compiled: _CompiledNetwork) -> tuple[FieldChannel, ...]:
        """Resolve source states once for both operator and compact mode solvers."""
        xp = select_array_module(contains_tracer(compiled.scattering))
        scattering = xp.asarray(compiled.scattering)
        fields = []
        for index, entry in enumerate(compiled.channels):
            exposure = entry.exposure
            occupation = (thermal_occupation_value(self._components[exposure._input_key[0]]._parameters)
                          if exposure._hidden else None)
            inbound = entry.reference.inbound
            if has_colored_noise(inbound):
                # Input j drives K_j = (S†L)_j; shared output support alone
                # cannot distinguish backaction from a downstream splitter.
                coupling = self._combine_maps(xp.conj(scattering[:, index]),
                                              [dict(c.coupling) for c in compiled.channels])
                if self._active_mapping_sources(coupling):
                    raise ValueError("A thermal reference filter feeds a quantum coupling; use an explicit "
                                     "dynamical filter/bath for colored thermal backaction.")
            elif any(source_occupation(e) is not None for e in inbound):
                # Filters before all occupied sources do not color their noise.
                flat = tuple(e for e in inbound if not isinstance(e, ReferenceFilter))
                occupation = noise_density(flat, 0.0, xp)
            fields.append(FieldChannel(exposure.label, entry.reference, occupation))
        return tuple(fields)

    def to_dict(self) -> dict[str, Any]:
        """Serialize a static network graph and its quantum ports."""
        filters = [component.label for component in self.components if component._transfer is not None]
        if filters:
            raise TypeError(
                f"PortNetwork filter components {sorted(filters)} use Python "
                "transfer callables and cannot be serialized by to_dict(). Chip.clone() and "
                "Chip.with_params() preserve them."
            )
        return {
            "label": self.label,
            "ports": [port.to_dict() for port in self._ports],
            "scattering": self._serialize_boundary(self._authored_scattering),
            "components": [
                self._serialize_component(component)
                for component in self.components
                if not any(port is not None for port in component._local_ports)
            ],
            "connections": [
                {
                    "output": list(output_key),
                    "input": list(input_key),
                }
                for input_key, output_key in self._connections.items()
            ],
            "exposures": [
                {
                    "label": exposure.label,
                    "input": list(exposure._input_key),
                    "output": list(exposure._output_key),
                }
                for exposure in self._exposures
            ],
        }

    def _serialize_component(self, component: SLHComponent) -> dict[str, Any]:
        kind = component._kind
        if kind == "scattering":
            return {
                "label": component.label,
                "kind": kind,
                "terminals": list(component.input_names),
                "scattering": self._serialize_matrix(component.scattering),
            }
        if kind == "circulator":
            parameters: dict[str, Any] = {"ports": len(component.sides)}
        elif kind == "permutation":
            parameters = {"order": [inputs[0] for inputs in component._row_inputs or ()]}
        else:
            parameters = {}
        parameters.update(
            (name, self._serialize_scalar(value))
            for name, value in component._parameters.items()
        )
        return {"label": component.label, "kind": kind, "parameters": parameters}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PortNetwork":
        """Reconstruct a static network produced by :meth:`to_dict`.

        Parameters
        ----------
        data : mapping
            Serialized network payload.
        """
        unknown = set(data) - {
            "label",
            "ports",
            "scattering",
            "components",
            "connections",
            "exposures",
        }
        if unknown:
            raise TypeError(f"Unsupported serialized PortNetwork fields: {sorted(unknown)}")
        network = cls(
            label=data.get("label"),
            scattering=cls._deserialize_boundary(data.get("scattering")),
        )
        for payload in data.get("ports", []):
            network._add_port(Port.from_dict(payload))
        for payload in data.get("components", []):
            if "kind" not in payload:
                raise TypeError(f"Serialized PortNetwork component {payload.get('label')!r} has no kind.")
            kind = payload["kind"]
            if kind == "scattering":
                network.component(
                    payload["label"],
                    scattering=cls._deserialize_matrix(payload["scattering"]),
                    terminals=tuple(payload["terminals"]),
                )
                continue
            if kind not in _SERIALIZED_FACTORIES:
                raise TypeError(f"Unknown serialized PortNetwork component kind {kind!r}.")
            parameters = {
                name: cls._deserialize_scalar(value) for name, value in payload.get("parameters", {}).items()
            }
            getattr(network, kind)(payload["label"], **parameters)
        for payload in data.get("connections", []):
            input_key = tuple(payload["input"])
            output_key = tuple(payload["output"])
            network._connections[input_key] = output_key
            network._used_outputs[output_key] = input_key
        for payload in data.get("exposures", []):
            network._exposures.append(
                NetworkPort(
                    payload["label"],
                    tuple(payload["input"]),
                    tuple(payload["output"]),
                    _network_token=network._token,
                )
            )
        return network

    def copy(self) -> "PortNetwork":
        """Return an independent structural copy with label-based port targets."""
        return self._copy_with_port_replacements({})

    def restrict(self, ports: Sequence[str | Port]) -> "PortNetwork":
        """Return an independent sub-network of field-graph components reachable from ``ports``.

        Every reachable component is copied with its connections, exposures, tracked
        parameters, and filter callables. Authored boundary scattering is limited to
        the kept planes. A reachable subgraph that also touches an unselected port
        raises, because cutting it would change the SLH dynamics.

        Parameters
        ----------
        ports : sequence of str or Port
            Quantum coupling ports whose field subgraphs are kept.

        Returns
        -------
        PortNetwork
            An independent network with label-based port targets.

        Raises
        ------
        KeyError
            A selected port is not part of this network.
        ValueError
            A reachable subgraph also touches an unselected port, or boundary
            scattering couples kept and dropped planes.
        """
        labels = {port if isinstance(port, str) else port.label for port in ports}
        known = {port.label for port in self._ports}
        missing = sorted(labels - known)
        if missing:
            raise KeyError(f"Unknown PortNetwork ports {missing}; available: {sorted(known)}.")
        neighbours: dict[str, set[str]] = {component.label: set() for component in self.components}
        edges = [*self._connections.items(), *((item._input_key, item._output_key) for item in self._exposures)]
        for first, second in edges:
            neighbours[first[0]].add(second[0])
            neighbours[second[0]].add(first[0])
        keep: set[str] = set()
        stack = list(labels)
        while stack:
            label = stack.pop()
            if label not in keep:
                keep.add(label)
                stack.extend(neighbours[label])
        stray = sorted(keep & known - labels)
        if stray:
            raise ValueError(
                f"PortNetwork subgraph reached from ports {sorted(labels)} also touches ports {stray}; "
                "restrict them together."
            )
        return self._copy_graph({}, keep)

    def include(self, template: "PortNetwork", *, prefix: str) -> IncludedNetwork:
        """Copy a reusable network template into this network under ``prefix``.

        Components and connections are copied with labels ``prefix/label``. Tracked
        parameters therefore use paths such as
        ``network.component.prefix/label.name``. Filter callables are retained, and
        the template remains unchanged so it can be included again under another
        prefix.

        The template's exposures become interfaces rather than host planes. Exposures
        made with ``expose(..., at=...)`` are available through ``side(name)``;
        asymmetric exposures use ``input(name)`` and ``output(name)``. Wire the
        returned interfaces with :meth:`link`, :meth:`cascade`, or :meth:`expose`.

        Parameters
        ----------
        template : PortNetwork
            A network without quantum ports or authored boundary scattering.
        prefix : str
            A label prefix for the copied components, unique within this network.

        Returns
        -------
        IncludedNetwork
            Accessors for the copied components and interfaces.

        Raises
        ------
        ValueError
            The template owns ports or boundary scattering, or ``prefix`` is already
            in use.
        """
        if template is self:
            raise ValueError("A PortNetwork cannot include itself.")
        if template._ports:
            raise ValueError("Included networks cannot own quantum ports; declare ports on the host network.")
        if template._authored_scattering is not None:
            raise ValueError("Included networks cannot author boundary scattering; declare it on the host network.")
        if any(label.startswith(f"{prefix}/") for label in self._components):
            raise ValueError(f"Prefix {prefix!r} is already used in this PortNetwork.")

        def rename(key: TerminalKey) -> TerminalKey:
            return (f"{prefix}/{key[0]}", key[1])

        for component in template.components:
            self._add_component(
                template._clone_component(component, self._token, label=f"{prefix}/{component.label}")
            )
        for input_key, output_key in template._connections.items():
            new_input, new_output = rename(input_key), rename(output_key)
            self._connections[new_input] = new_output
            self._used_outputs[new_output] = new_input
        interfaces = {
            exposure.label: (rename(exposure._input_key), rename(exposure._output_key))
            for exposure in template._exposures
        }
        return IncludedNetwork(prefix, self, MappingProxyType(interfaces))

    @staticmethod
    def _clone_component(component: SLHComponent, token: object, *, label: str | None = None) -> SLHComponent:
        """Return a portless component copy bound to ``token``, optionally relabelled."""
        new_label = component.label if label is None else label
        return replace(
            component,
            label=new_label,
            scattering=copy_value(component.scattering),
            _parameters=copy_value(component._parameters),
            _network_token=token,
            _hidden_pairs=tuple(
                (first, second, f"hidden.{new_label}.{first}") for first, second, _ in component._hidden_pairs
            ),
        )

    def _copy_with_port_replacements(
        self,
        replacements: Mapping[str, Port],
    ) -> "PortNetwork":
        """Copy this graph while replacing selected quantum coupling ports."""
        known = {port.label for port in self._ports}
        unknown = set(replacements) - known
        if unknown:
            raise KeyError(f"Cannot replace unknown PortNetwork ports: {sorted(unknown)}.")
        for label, replacement in replacements.items():
            if replacement.label != label:
                raise ValueError(
                    f"Replacement for port {label!r} must retain that label, got {replacement.label!r}."
                )
        return self._copy_graph(replacements, None)

    def _restricted_boundary(self, keep: set[str]) -> Any:
        """Return the authored boundary scattering limited to planes inside ``keep``."""
        authored = self._authored_scattering
        if authored is None:
            return None
        planes = [exposure for exposure in self._effective_exposures() if not exposure._hidden]
        kept = {exposure.label for exposure in planes if exposure._input_key[0] in keep}
        if isinstance(authored, Mapping):
            restricted = {}
            for (output, input_), value in authored.items():
                has_output, has_input = resolve_label(output) in kept, resolve_label(input_) in kept
                if has_output != has_input:
                    raise ValueError(
                        f"Boundary scattering entry {(resolve_label(output), resolve_label(input_))} couples "
                        "planes of different field subgraphs."
                    )
                if has_output:
                    restricted[(output, input_)] = copy_value(value)
            return restricted or None
        if len(kept) == len(planes):
            return copy_value(authored)
        if not kept:
            return None
        raise ValueError("Matrix-form boundary scattering spans planes of different field subgraphs.")

    def _copy_graph(self, replacements: Mapping[str, Port], keep: set[str] | None) -> "PortNetwork":
        """Copy the components in ``keep`` (all when ``None``) with their wiring and parameters."""
        kept = keep if keep is not None else {component.label for component in self.components}
        scattering = copy_value(self._authored_scattering) if keep is None else self._restricted_boundary(kept)
        copied = PortNetwork(scattering=scattering, label=self.label)
        for component in self.components:
            if component.label not in kept:
                continue
            if any(port is not None for port in component._local_ports):
                assert len(component._local_ports) == 1
                local_port = component._local_ports[0]
                assert local_port is not None
                copied._add_port(replacements.get(local_port.label, local_port).copy())
            else:
                copied._add_component(self._clone_component(component, copied._token))
        copied._connections = {
            input_key: output_key for input_key, output_key in self._connections.items() if input_key[0] in kept
        }
        copied._used_outputs = {output_key: input_key for input_key, output_key in copied._connections.items()}
        copied._exposures = [
            replace(exposure, _network_token=copied._token)
            for exposure in self._exposures
            if exposure._input_key[0] in kept
        ]
        return copied

    def physics_notes(self) -> list[str]:
        """Return the boundary convention and approximation scope."""
        return [
            "This instantaneous Markovian boundary uses b_out = S b_in + L; scalar S is unitary after "
            "explicit loss dilation; declared thermal input states enter the quantum dynamics.",
            "Reference sections shift incident and reported fields at external planes only; they do not "
            "enter S, L, or H and cannot sit inside feedback loops. Acyclic output mixing preserves joint noise.",
        ]

    def dynamical_supports(
        self, chip: Any, *, _compiled: _CompiledNetwork | None = None,
    ) -> tuple[tuple[str, ...], ...]:
        """Return device groups coupled by cascade-generated Hamiltonian terms.

        Static scattering alone is excluded: a unitary mixer changes field
        coordinates but leaves the summed Lindbladian invariant. Each
        downstream/upstream coupling pair generated by SLH series composition
        contributes the union of its two quantum-port supports.

        Parameters
        ----------
        chip : Chip
            Chip used to resolve port target supports.
        """
        supports = {
            port.label: port.resolve_targets(chip)
            for port in self._ports
        }
        groups: list[tuple[str, ...]] = []
        compiled = self._compile() if _compiled is None else _compiled
        for downstream, upstream, _coefficient in self._active_pairs(compiled.generated_pairs):
            touched = set(supports[downstream] + supports[upstream])
            members = tuple(device.label for device in chip.devices if device.label in touched)
            if len(members) > 1 and members not in groups:
                groups.append(members)
        return tuple(groups)

    def fed_ports(
        self, chip: Any, exposure: str, *, _compiled: _CompiledNetwork | None = None,
    ) -> list[tuple[Any, Any]]:
        """Return ``(port, coefficient)`` pairs reached from ``exposure``.

        Each coefficient combines the scattering entry from the exposure to a
        channel with that port's weight in the channel. A coherent input ``β``
        therefore reaches the port coupling as ``c = coefficient · β``.

        Parameters
        ----------
        chip : Chip
            Chip used to resolve the network boundary.
        exposure : str
            External input reference-plane label.
        """
        compiled = self._compile() if _compiled is None else _compiled
        labels = [item.exposure.label for item in compiled.channels]
        if exposure not in labels:
            raise ValueError(f"Unknown resolved port {exposure!r}.")
        column = labels.index(exposure)
        fed: list[tuple[Any, Any]] = []
        for row, channel in enumerate(compiled.channels):
            if not compiled.support[row, column]:
                continue
            mapping = channel.coupling
            gain = compiled.scattering[row][column]
            for source in self._active_mapping_sources(mapping):
                fed.append((chip.port(source), gain * mapping[source]))
        return fed

    @staticmethod
    def _active_pairs(
        pairs: Sequence[tuple[str, str, Any]],
    ) -> tuple[tuple[str, str, Any], ...]:
        """Drop only generated Hamiltonian pairs known to have zero weight."""
        return tuple(
            (downstream, upstream, coefficient)
            for downstream, upstream, coefficient in pairs
            if not PortNetwork._is_concrete_zero(coefficient)
        )

    def _active_generated_pairs(self) -> tuple[tuple[str, str, Any], ...]:
        """Return quantum-port pairs that generate a series Hamiltonian."""
        return self._active_pairs(self._compile().generated_pairs)

    @staticmethod
    def _serialize_matrix(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if contains_tracer(value):
            raise TypeError("Traced PortNetwork scattering cannot be serialized.")
        matrix = np.asarray(value, dtype=complex)
        return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}

    @staticmethod
    def _deserialize_matrix(value: Mapping[str, Any] | None) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value["real"]) + 1j * np.asarray(value["imag"])

    @classmethod
    def _serialize_boundary(cls, value: Any) -> Any:
        if value is None:
            return None
        if callable(value):
            raise TypeError("Callable PortNetwork scattering is not serializable.")
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "entries": [
                    {
                        "output": resolve_label(output),
                        "input": resolve_label(input_),
                        "value": cls._serialize_scalar(item),
                    }
                    for (output, input_), item in value.items()
                ],
            }
        return {"kind": "matrix", "value": cls._serialize_matrix(value)}

    @classmethod
    def _deserialize_boundary(cls, value: Any) -> Any:
        if value is None:
            return None
        if value["kind"] == "matrix":
            return cls._deserialize_matrix(value["value"])
        if value["kind"] == "mapping":
            return {
                (entry["output"], entry["input"]): cls._deserialize_scalar(entry["value"])
                for entry in value["entries"]
            }
        raise TypeError(f"Unknown serialized PortNetwork scattering kind {value['kind']!r}.")

    @staticmethod
    def _serialize_scalar(value: Any) -> dict[str, float]:
        if contains_tracer(value):
            raise TypeError("Traced PortNetwork values cannot be serialized.")
        scalar = complex(np.asarray(value).item())
        return {"real": scalar.real, "imag": scalar.imag}

    @staticmethod
    def _deserialize_scalar(value: Any) -> Any:
        # Structural options such as a permutation order or a port count are
        # plain JSON values and pass through untouched.
        if not isinstance(value, Mapping):
            return value
        scalar = complex(value["real"], value["imag"])
        return scalar.real if scalar.imag == 0.0 else scalar

    def _add_port(self, port: Port) -> None:
        if not isinstance(port, Port):
            raise TypeError(f"Expected a Port, got {type(port).__name__}: {port!r}")
        component = SLHComponent(
            label=port.label,
            input_names=("field",),
            output_names=("field",),
            scattering=np.asarray([[1.0]], dtype=complex),
            _network_token=self._token,
            sides=("field",),
            _local_ports=(port,),
            _kind="port",
        )
        self._add_component(component)
        port._bind_network(self, component)
        self._ports.append(port)

    def _add_component(self, component: SLHComponent) -> SLHComponent:
        if component.label in self._components:
            raise ValueError(f"Duplicate PortNetwork component label {component.label!r}.")
        self._components[component.label] = component
        return component

    def _effective_exposures(self) -> tuple[NetworkPort, ...]:
        exposed = list(self._exposures)
        used = {
            key
            for exposure in self._exposures
            for key in (exposure._input_key, exposure._output_key)
        }
        for component in self.components:
            if not any(port is not None for port in component._local_ports):
                continue
            input_key = (component.label, component.input_names[0])
            output_key = (component.label, component.output_names[0])
            touched = (
                input_key in self._connections
                or output_key in self._used_outputs
                or input_key in used
                or output_key in used
            )
            if not touched:
                exposed.append(
                    NetworkPort(
                        component.label,
                        input_key,
                        output_key,
                        _network_token=self._token,
                    )
                )
        hidden: list[NetworkPort] = []
        for component in self.components:
            if self._is_reference(component.label):
                continue
            for input_name, output_name, label in component._hidden_pairs:
                hidden.append(
                    NetworkPort(
                        label,
                        (component.label, input_name),
                        (component.label, output_name),
                        _hidden=True,
                        _network_token=self._token,
                    )
                )
        return tuple((*exposed, *hidden))

    def _is_reference(self, label: str) -> bool:
        kind = self._components[label]._kind
        if kind in {"delay", "filter", "amplifier"}:
            return True
        # A passive two-sided section downstream of a reference run belongs
        # to that same external run. Its directed vacuum/thermal dilation is
        # retained in ReferenceLoss rather than becoming an internal amplifier.
        visited: set[str] = set()
        while kind in {"attenuator", "isolator"} and label not in visited:
            visited.add(label)
            previous = self._connections.get((label, "1"))
            if previous is None or previous[1] != "2":
                return False
            label = previous[0]
            kind = self._components[label]._kind
            if kind in {"delay", "filter", "amplifier"}:
                return True
        return False

    def _reference_element(self, label: str, *, output_side: str | None = None) -> ReferenceElement:
        component = self._components[label]
        parameters = component._parameters
        kind = component._kind
        if kind == "delay":
            return ReferenceDelay(label, parameters["duration"])
        if kind == "amplifier":
            return ReferenceAmplifier(label, *amplifier_values(parameters))
        if kind in {"attenuator", "isolator"}:
            assert output_side is not None
            eta = attenuation_value(parameters) if kind == "attenuator" else float(output_side == "2")
            return ReferenceLoss(label, eta, thermal_occupation_value(parameters))
        assert component._transfer is not None
        noise = thermal_occupation_value(parameters)
        transfer_parameters = {name: value for name, value in parameters.items()
                               if name != "thermal_occupation"}
        return ReferenceFilter(label, component._transfer, MappingProxyType(transfer_parameters),
                               noise)

    def _peel(
        self, exposure: NetworkPort, *, traversals: set[TerminalKey] | None = None,
    ) -> tuple[TerminalKey, TerminalKey, ReferencePlane]:
        """Peel adjacent reference runs from ``exposure`` to the Markov boundary.

        The returned plane orders inbound sections from exposure to boundary and
        outbound sections from boundary to exposure.
        """

        def walk(
            key: TerminalKey, step: Mapping[TerminalKey, TerminalKey], *, outbound: bool
        ) -> tuple[TerminalKey, list[ReferenceElement]]:
            run: list[ReferenceElement] = []
            while self._is_reference(key[0]):
                label, name = key
                if traversals is not None:
                    traversals.add((label, name if outbound else ("2" if name == "1" else "1")))
                if any(element.label == label for element in run):
                    raise ValueError(f"Reference component {label!r} feeds back into itself.")
                if self._components[label]._kind == "amplifier":
                    if name != "2":
                        raise ValueError(
                            f"Amplifier {label!r} must lie on an output line with side 1 toward the "
                            "chip and side 2 toward the exposed plane; it cannot amplify an incident "
                            "field into the chip."
                        )
                    if outbound:
                        run.append(self._reference_element(label))
                elif self._components[label]._kind in {"attenuator", "isolator"}:
                    forward = (name == "2") if outbound else (name == "1")
                    run.append(self._reference_element(label, output_side="2" if forward else "1"))
                else:
                    run.append(self._reference_element(label))
                other = (label, "2" if name == "1" else "1")
                if other not in step:
                    raise ValueError(
                        f"Reference component {label!r} must connect toward the Markov boundary; "
                        f"terminal {other} is free."
                    )
                key = step[other]
            return key, run

        boundary_input, inbound = walk(exposure._input_key, self._used_outputs, outbound=False)
        boundary_output, outbound = walk(exposure._output_key, self._connections, outbound=True)
        plane = ReferencePlane(inbound=tuple(inbound), outbound=tuple(reversed(outbound)))
        return boundary_input, boundary_output, plane

    def _compile(
        self, *, _embedded: Mapping[TerminalKey, ReferenceElement] | None = None,
        _retained: Mapping[str, ReferencePlane] | None = None,
    ) -> _CompiledNetwork:
        exposures = self._effective_exposures()
        traversals: set[TerminalKey] = set()
        peeled = [self._peel(exposure, traversals=traversals) for exposure in exposures]
        planes = [(_retained or {}).get(exposure.label, plane)
                  for exposure, (_, _, plane) in zip(exposures, peeled, strict=True)]
        def active_sides(label: str) -> tuple[str, ...]:
            return tuple(side for side in ("1", "2") if (label, side) in traversals
                         or (label, side) in self._used_outputs
                         or (label, "2" if side == "1" else "1") in self._connections)

        unused = [label for label in self._components if self._is_reference(label) and not active_sides(label)]
        if unused:
            raise ValueError(f"Reference components {unused} must connect to an exposed run "
                             "or downstream output graph.")
        internal = [label for label in self._components if self._is_reference(label)
                    and any((label, side) not in traversals for side in active_sides(label))]
        if internal:
            if _embedded is not None:
                raise ValueError("Reference graph cannot be reduced to exposed downstream fields.")
            # A component can have an exposed inbound leg and a branched outbound
            # leg. Preserve each traversal separately when forming the unitary core.
            from copy import copy
            bypassed = copy(self)
            bypassed._components = dict(self._components)
            embedded: dict[TerminalKey, ReferenceElement] = {}
            for label in internal:
                component = self._components[label]
                sides = active_sides(label)
                for side in sides:
                    if (label, side) in traversals or (component._kind == "amplifier" and side == "1"):
                        continue
                    if component._kind in {"isolator", "attenuator"}:
                        element = self._reference_element(label, output_side=side)
                    else:
                        element = self._reference_element(label)
                    embedded[(label, side)] = element
                bypassed._components[label] = replace(
                    component, _kind="through", _parameters={}, _transfer=None,
                    input_names=tuple("2" if side == "1" else "1" for side in sides), output_names=sides,
                    scattering=np.eye(len(sides), dtype=complex), _row_inputs=tuple((i,) for i in range(len(sides))),
                    _local_ports=(None,)*len(sides), _hidden_pairs=(), sides=())
            retained = {exposure.label: plane for exposure, plane in zip(exposures, planes, strict=True)}
            return bypassed._compile(_embedded=embedded, _retained=retained)
        core = [component for component in self.components if not self._is_reference(component.label)]
        covered_inputs = {boundary_input for boundary_input, _, _ in peeled}
        covered_outputs = {boundary_output for _, boundary_output, _ in peeled}
        all_inputs = {(component.label, name) for component in core for name in component.input_names}
        all_outputs = {(component.label, name) for component in core for name in component.output_names}
        free_inputs = all_inputs - set(self._connections) - covered_inputs
        free_outputs = all_outputs - set(self._used_outputs) - covered_outputs
        if free_inputs or free_outputs:
            raise ValueError(
                (f"Reference components {list(_embedded)} must be outside a feedback loop with all terminals covered. "
                 if _embedded else "") + "PortNetwork has free terminals; connect or expose them explicitly: "
                f"inputs={sorted(free_inputs)}, outputs={sorted(free_outputs)}."
            )
        if len(covered_inputs) != len(exposures) or len(covered_outputs) != len(exposures):
            raise ValueError("PortNetwork exposures must use distinct input and output terminals.")

        size = len(exposures)
        input_fields: dict[TerminalKey, _AffineField] = {}
        output_fields: dict[TerminalKey, _AffineField] = {}
        for column, (boundary_input, _, _) in enumerate(peeled):
            basis = [0.0] * size
            basis[column] = 1.0
            input_fields[boundary_input] = _AffineField(basis, {}, [index == column for index in range(size)])

        nodes = {
            (component.label, name): (component, row)
            for component in core
            for row, name in enumerate(component.output_names)
        }
        matrices = {component.label: self._component_matrix(component) for component in core}

        def _feeders(terminal: TerminalKey) -> list[_Feeder]:
            """Return ``(coefficient, boundary_input, upstream_output)`` per structurally fed column."""
            component, row = nodes[terminal]
            columns = (
                range(len(component.input_names)) if component._row_inputs is None else component._row_inputs[row]
            )
            result: list[_Feeder] = []
            for column in columns:
                key = (component.label, component.input_names[column])
                coefficient = matrices[component.label][row, column]
                if key in input_fields:
                    result.append((coefficient, key, None))
                else:
                    result.append((coefficient, None, self._connections[key]))
            return result

        feeds = {terminal: _feeders(terminal) for terminal in nodes}
        generated_pairs: list[tuple[str, str, Any]] = []
        groups = []
        for members in _strongly_connected(nodes, lambda terminal: [up for _, _, up in feeds[terminal] if up]):
            cyclic = len(members) > 1 or any(up == members[0] for _, _, up in feeds[members[0]])
            groups.append((members, cyclic))
            if cyclic:
                solved = self._solve_loop(members, nodes, feeds, input_fields, output_fields, size)
            else:
                solved = {members[0]: self._propagate(members[0], nodes, feeds, input_fields, output_fields, size)}
            for terminal, solved_field in solved.items():
                component, row = nodes[terminal]
                local_port = component._local_ports[row]
                if local_port is not None:
                    upstream = dict(solved_field.coupling)
                    upstream[local_port.label] = upstream.get(local_port.label, 0.0) - 1.0
                    generated_pairs.extend(
                        (local_port.label, source, coefficient)
                        for source, coefficient in upstream.items()
                        if not self._is_concrete_zero(coefficient)
                    )
                output_fields[terminal] = solved_field

        rows = [output_fields[boundary_output] for _, boundary_output, _ in peeled]
        support_matrix = np.asarray([row.support for row in rows], dtype=bool)
        scattering_rows = [row.scattering for row in rows]
        xp = select_array_module(contains_tracer(scattering_rows))
        scattering = xp.asarray(scattering_rows, dtype=complex)
        coupling_maps = [row.coupling for row in rows]

        external_count = sum(not exposure._hidden for exposure in exposures)
        boundary = self._boundary_scattering(tuple(exposure.label for exposure in exposures[:external_count]))
        if boundary is not None:
            boundary_xp = select_array_module(contains_tracer((scattering, boundary)))
            boundary = boundary_xp.asarray(boundary, dtype=complex)
            if boundary.shape != (external_count, external_count):
                raise ValueError(
                    "PortNetwork scattering shape must match exposed channels; "
                    f"got {boundary.shape} for {external_count}."
                )
            full_boundary = boundary_xp.eye(size, dtype=complex)
            if hasattr(full_boundary, "at"):
                full_boundary = full_boundary.at[:external_count, :external_count].set(boundary)
            else:
                full_boundary[:external_count, :external_count] = boundary
            scattering = full_boundary @ scattering
            coupling_maps = [
                self._combine_maps(full_boundary[row], coupling_maps)
                for row in range(size)
            ]
            mixes = np.asarray(
                [[not self._is_concrete_zero(full_boundary[row, k]) for k in range(size)] for row in range(size)],
                dtype=bool,
            )
            support_matrix = (mixes.astype(int) @ support_matrix.astype(int)) > 0

        self._validate_unitary(scattering)
        return _CompiledNetwork(
            channels=tuple(
                _CompiledChannel(exposure, MappingProxyType(mapping), plane)
                for exposure, mapping, plane in zip(exposures, coupling_maps, planes, strict=True)
            ),
            scattering=scattering,
            generated_pairs=tuple(generated_pairs),
            support=support_matrix,
            output_network=(self._output_network(_embedded, groups, nodes, feeds, output_fields,
                                                 input_fields, peeled, scattering, boundary)
                            if _embedded else None),
        )

    @staticmethod
    def _output_network(embedded, groups, nodes, feeds, fields, inputs, peeled, scattering, boundary):
        """Freeze acyclic downstream field maps, preserving a unitary solver boundary."""
        from quchip.engine.output_network import OutputNetwork, OutputStep, pad_mixing
        xp = select_array_module(contains_tracer(scattering))
        inverse = xp.conj(xp.asarray(scattering).T)
        affected = set()
        steps = []
        for members, cyclic in groups:
            local = {}
            for terminal in members:
                reference = embedded.get(terminal)
                changed = reference is not None or any(upstream in affected for _, _, upstream in feeds[terminal])
                local[terminal] = (reference, changed)
            if cyclic and any(changed for _, changed in local.values()):
                raise ValueError(f"Reference components {sorted({key[0] for key in embedded})} must remain "
                                 "outside feedback loops; "
                                 "only acyclic downstream scattering is supported.")
            for terminal in members:
                reference, changed = local[terminal]
                component, row = nodes[terminal]
                if changed and component._local_ports[row] is not None:
                    raise ValueError("An internal reference component feeds a quantum coupling; "
                                     "place amplifiers on output lines and use a dynamical model for internal filters.")
                base = xp.asarray(fields[terminal].scattering) @ inverse
                terms = ()
                if changed:
                    affected.add(terminal)
                    terms = tuple((coefficient, upstream,
                                   xp.asarray(inputs[boundary_input].scattering) @ inverse
                                   if boundary_input is not None else None)
                                  for coefficient, boundary_input, upstream in feeds[terminal])
                steps.append(OutputStep(terminal, base, terms, reference))
        size = len(scattering)
        full_boundary = xp.eye(size, dtype=complex) if boundary is None else pad_mixing(boundary, size, xp)
        return OutputNetwork(tuple(steps), tuple(output for _, output, _ in peeled), full_boundary, size)

    @staticmethod
    def _incoming(
        terminal: TerminalKey,
        feeds: Mapping[TerminalKey, list[_Feeder]],
        input_fields: Mapping[TerminalKey, _AffineField],
        output_fields: Mapping[TerminalKey, _AffineField],
        pending: Collection[TerminalKey] = (),
    ) -> tuple[list[tuple[Any, _AffineField]], list[tuple[Any, TerminalKey]]]:
        """Split a terminal's feeders into known fields and unsolved loop members."""
        known: list[tuple[Any, _AffineField]] = []
        unsolved: list[tuple[Any, TerminalKey]] = []
        for coefficient, boundary, upstream in feeds[terminal]:
            if boundary is not None:
                known.append((coefficient, input_fields[boundary]))
                continue
            assert upstream is not None
            if upstream in pending:
                unsolved.append((coefficient, upstream))
            else:
                known.append((coefficient, output_fields[upstream]))
        return known, unsolved

    def _propagate(
        self,
        terminal: TerminalKey,
        nodes: Mapping[TerminalKey, tuple[SLHComponent, int]],
        feeds: Mapping[TerminalKey, list[_Feeder]],
        input_fields: Mapping[TerminalKey, _AffineField],
        output_fields: Mapping[TerminalKey, _AffineField],
        size: int,
    ) -> _AffineField:
        """Evaluate one output terminal whose feeders are all known."""
        known, _ = self._incoming(terminal, feeds, input_fields, output_fields)
        scattering = [
            sum(coefficient * known_field.scattering[index] for coefficient, known_field in known)
            for index in range(size)
        ]
        coupling: dict[str, Any] = {}
        for coefficient, known_field in known:
            for source, value in known_field.coupling.items():
                coupling[source] = coupling.get(source, 0.0) + coefficient * value
        support = [any(known_field.support[index] for _, known_field in known) for index in range(size)]
        component, row = nodes[terminal]
        local_port = component._local_ports[row]
        if local_port is not None:
            coupling[local_port.label] = coupling.get(local_port.label, 0.0) + 1.0
        return _AffineField(scattering, coupling, support)

    def _solve_loop(
        self,
        members: list[TerminalKey],
        nodes: Mapping[TerminalKey, tuple[SLHComponent, int]],
        feeds: Mapping[TerminalKey, list[_Feeder]],
        input_fields: Mapping[TerminalKey, _AffineField],
        output_fields: Mapping[TerminalKey, _AffineField],
        size: int,
    ) -> dict[TerminalKey, _AffineField]:
        """Solve the algebraic loop ``y = M y + b`` for one strongly connected set of terminals."""
        pending = set(members)
        position = {terminal: index for index, terminal in enumerate(members)}
        count = len(members)
        loop_matrix: list[list[Any]] = [[0.0] * count for _ in members]
        structure = np.zeros((count, count), dtype=bool)
        knowns: list[list[tuple[Any, _AffineField]]] = []
        for terminal in members:
            known, unsolved = self._incoming(terminal, feeds, input_fields, output_fields, pending)
            knowns.append(known)
            for coefficient, upstream in unsolved:
                loop_matrix[position[terminal]][position[upstream]] += coefficient
                structure[position[terminal], position[upstream]] = True
        sources: list[str] = []
        for index, terminal in enumerate(members):
            component, row = nodes[terminal]
            local_port = component._local_ports[row]
            candidates = [source for _, known_field in knowns[index] for source in known_field.coupling]
            if local_port is not None:
                candidates.append(local_port.label)
            sources.extend(source for source in candidates if source not in sources)
        width = size + len(sources)
        drive: list[list[Any]] = [[0.0] * width for _ in members]
        drive_support = np.zeros((count, size), dtype=bool)
        for index, terminal in enumerate(members):
            for coefficient, known_field in knowns[index]:
                for column in range(size):
                    drive[index][column] += coefficient * known_field.scattering[column]
                for source, value in known_field.coupling.items():
                    drive[index][size + sources.index(source)] += coefficient * value
                drive_support[index] |= np.asarray(known_field.support, dtype=bool)
            component, row = nodes[terminal]
            local_port = component._local_ports[row]
            if local_port is not None:
                drive[index][size + sources.index(local_port.label)] += 1.0
        xp = select_array_module(contains_tracer((loop_matrix, drive)))
        system = xp.eye(count, dtype=complex) - xp.asarray(loop_matrix, dtype=complex)
        if not contains_tracer(system):
            singular_values = np.linalg.svd(np.asarray(system), compute_uv=False)
            if singular_values[-1] <= 1e-12 * max(1.0, float(singular_values[0])):
                labels = sorted({terminal[0] for terminal in members})
                raise ValueError(f"Instantaneous feedback loop through {labels} is singular: I - M has no inverse.")
        solution = xp.linalg.solve(system, xp.asarray(drive, dtype=complex))
        support = drive_support.copy()
        for _ in members:
            support = drive_support | ((structure.astype(int) @ support.astype(int)) > 0)
        fields: dict[TerminalKey, _AffineField] = {}
        for index, terminal in enumerate(members):
            coupling = {
                source: solution[index, size + offset]
                for offset, source in enumerate(sources)
                if not self._is_concrete_zero(solution[index, size + offset])
            }
            fields[terminal] = _AffineField(
                [solution[index, column] for column in range(size)], coupling, [bool(flag) for flag in support[index]]
            )
        return fields

    def _boundary_scattering(self, labels: tuple[str, ...]) -> Any | None:
        authored = self._authored_scattering
        if authored is None:
            return None
        if callable(authored):
            raise TypeError("PortNetwork scattering is instantaneous; callables are not supported.")
        if isinstance(authored, Mapping):
            known = set(labels)
            rows: list[list[Any]] = []
            normalized: dict[tuple[str, str], Any] = {}
            for key, value in authored.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise TypeError("Scattering mappings use (output, input) keys.")
                output, input_ = (resolve_label(item) for item in key)
                if output not in known or input_ not in known:
                    raise ValueError(
                        f"Scattering entry {(output, input_)} references outside exposures {list(labels)}."
                    )
                normalized[(output, input_)] = value
            for output in labels:
                rows.append([normalized.get((output, input_), 0.0) for input_ in labels])
            return rows
        return authored

    @staticmethod
    def _combine_maps(coefficients: Any, maps: list[dict[str, Any]]) -> dict[str, Any]:
        combined: dict[str, Any] = {}
        for coefficient, mapping in zip(coefficients, maps, strict=True):
            for source, value in mapping.items():
                combined[source] = combined.get(source, 0.0) + coefficient * value
        return combined

    @staticmethod
    def _active_mapping_sources(mapping: Mapping[str, Any]) -> tuple[str, ...]:
        """Return sources whose scalar coefficient is not concretely zero."""
        return tuple(
            source
            for source, coefficient in mapping.items()
            if not PortNetwork._is_concrete_zero(coefficient)
        )

    @staticmethod
    def _same_frame_frequency(first: Any, second: Any) -> bool:
        """Return whether two port carriers are statically known to coincide."""
        first_concrete = maybe_concrete_scalar(first)
        second_concrete = maybe_concrete_scalar(second)
        if first_concrete is not None and second_concrete is not None:
            return bool(np.isclose(first_concrete, second_concrete, rtol=1e-12, atol=1e-12))
        return first is second

    @classmethod
    def _common_frame_frequency(
        cls,
        mapping: Mapping[str, Any],
        channels: Mapping[str, Any],
        *,
        boundary: str,
    ) -> Any:
        """Return one carrier for a statically composed output channel; ``None`` without coupling."""
        sources = cls._active_mapping_sources(mapping)
        if not sources:
            return None
        reference = channels[sources[0]].collapse.frame_frequency
        if any(
            not cls._same_frame_frequency(
                reference,
                channels[source].collapse.frame_frequency,
            )
            for source in sources[1:]
        ):
            raise ValueError(
                f"PortNetwork cannot statically compose {boundary} from ports {list(sources)}: their "
                "rotating-frame frequencies are not statically known to be equal (traced frequencies must "
                "share the same traced value). Use the lab/common frame; time-dependent collapse channels "
                "are not implemented."
            )
        return reference

    @staticmethod
    def _materialize_coupling(
        mapping: Mapping[str, Any],
        operators: dict[str, Any],
        *,
        key: str,
    ) -> Any:
        from quchip.engine.ir import CanonicalOperator

        if not operators:
            raise RuntimeError("A PortNetwork cannot resolve without quantum coupling ports.")
        first = next(iter(operators.values()))
        prefer_jax = contains_tracer((
            tuple(mapping.values()), tuple(operator.values for operator in operators.values())
        ))
        xp = select_array_module(prefer_jax)
        values = xp.zeros(first.shape, dtype=complex)
        for source, coefficient in mapping.items():
            values = values + xp.asarray(coefficient) * xp.asarray(operators[source].to_dense())
        return CanonicalOperator.from_dense(
            values,
            dims=first.dims,
            basis=first.basis,
            subsystem_labels=first.subsystem_labels,
            tag=f"slh:{key}",
        )

    def _materialize_generated_hamiltonian(
        self,
        pairs: Sequence[tuple[str, str, Any]],
        operators: dict[str, Any],
    ) -> Any | None:
        from quchip.engine.ir import CanonicalOperator

        if not pairs:
            return None
        first = next(iter(operators.values()))
        prefer_jax = contains_tracer((pairs, tuple(operator.values for operator in operators.values())))
        xp = select_array_module(prefer_jax)
        values = xp.zeros(first.shape, dtype=complex)
        for downstream, upstream, coefficient in pairs:
            left = xp.asarray(operators[downstream].to_dense())
            right = xp.asarray(operators[upstream].to_dense())
            product = xp.conj(xp.swapaxes(left, -1, -2)) @ (xp.asarray(coefficient) * right)
            values = values + (product - xp.conj(xp.swapaxes(product, -1, -2))) / (2j)
        if not contains_tracer(values) and np.allclose(np.asarray(values), 0.0):
            return None
        return CanonicalOperator.from_dense(
            values,
            dims=first.dims,
            basis=first.basis,
            subsystem_labels=first.subsystem_labels,
            tag=f"slh-network:{self.label}",
        )

    def _component_matrix(self, component: SLHComponent) -> Any:
        kind = component._kind
        parameters = component._parameters
        if kind == "phase_shift":
            phase = parameters["phase"]
            xp = select_array_module(contains_tracer(phase))
            matrix = xp.asarray([[xp.exp(1j * xp.asarray(phase))]])
        elif kind == "beam_splitter":
            matrix = self._transmission_matrix(parameters["eta"])
        elif kind == "attenuator":
            matrix = self._attenuator_matrix(attenuation_value(parameters))
        else:
            matrix = component.scattering
        xp = select_array_module(contains_tracer(matrix))
        matrix = xp.asarray(matrix, dtype=complex)
        expected = (len(component.output_names), len(component.input_names))
        if matrix.shape != expected:
            raise ValueError(
                f"Component {component.label!r} scattering shape {matrix.shape} must be {expected}."
            )
        self._validate_unitary(matrix)
        return matrix

    @staticmethod
    def _validate_unitary(matrix: Any) -> None:
        if contains_tracer(matrix):
            return
        concrete = np.asarray(matrix, dtype=complex)
        identity = np.eye(concrete.shape[0], dtype=complex)
        if not np.allclose(concrete.conj().T @ concrete, identity, rtol=1e-10, atol=1e-12):
            raise ValueError("Concrete PortNetwork scattering must be unitary.")

    @staticmethod
    def _transmission_matrix(eta: Any) -> Any:
        """Return ``[[t, r], [-r, t]]`` for the directional splitter.

        Here ``t = sqrt(eta)`` and ``r = sqrt(1 - eta)``. Rows select outputs
        and columns select inputs, so ``eta`` is the power from input ``k`` to
        output ``k``.
        """
        xp = select_array_module(contains_tracer(eta))
        transmission = xp.sqrt(xp.asarray(eta))
        loss = xp.sqrt(1.0 - xp.asarray(eta))
        return xp.asarray([[transmission, loss], [-loss, transmission]], dtype=complex)

    @staticmethod
    def _attenuator_matrix(eta: Any) -> Any:
        """Return the reciprocal unitary dilation for a two-sided attenuator."""
        xp = select_array_module(contains_tracer(eta))
        t = xp.sqrt(xp.asarray(eta))
        r = xp.sqrt(1.0 - xp.asarray(eta))
        return xp.asarray(
            [[0.0, t, 0.0, r], [t, 0.0, r, 0.0], [-r, 0.0, t, 0.0], [0.0, -r, 0.0, t]],
            dtype=complex,
        )

    def _validate_terminal(self, terminal: FieldTerminal, direction: TerminalDirection) -> None:
        if not isinstance(terminal, FieldTerminal):
            raise TypeError(f"Expected a FieldTerminal, got {type(terminal).__name__}.")
        if terminal._network_token is not self._token:
            raise ValueError("Cannot connect terminals from different PortNetwork objects.")
        if terminal.direction != direction:
            raise ValueError(f"Expected an {direction} terminal, got {terminal.direction}.")
        component = self._components.get(terminal.component)
        if component is None:
            raise ValueError(f"Unknown component {terminal.component!r}.")
        names = component.input_names if direction == "input" else component.output_names
        if terminal.name not in names:
            raise ValueError(f"Unknown terminal {terminal.key}.")

    def _reject_hidden_terminal(self, terminal: FieldTerminal) -> None:
        component = self._components[terminal.component]
        hidden_names = {
            input_name if terminal.direction == "input" else output_name
            for input_name, output_name, _ in component._hidden_pairs
        }
        if terminal.name in hidden_names:
            raise ValueError("Vacuum-dilation terminals are network-owned and cannot be exposed or rewired.")

    def _terminal_is_exposed(self, terminal: FieldTerminal) -> bool:
        return any(
            terminal.key == (exposure._input_key if terminal.direction == "input" else exposure._output_key)
            for exposure in self._exposures
        )

    def _component_for_port(self, port: Port) -> SLHComponent:
        for component in self.components:
            if any(candidate is port for candidate in component._local_ports):
                return component
        raise ValueError(f"Port {port.label!r} is not owned by this PortNetwork.")

    def _side_of(
        self, value: Port | SLHComponent | ComponentPort, *, leaving: bool | None
    ) -> ComponentPort:
        """Resolve a component port for ``link`` or ``expose``.

        For a two-sided component, ``leaving=True`` selects side 2,
        ``leaving=False`` selects side 1, and ``leaving=None`` requires an
        explicit side.
        """
        if isinstance(value, ComponentPort):
            self._validate_terminal(value.input, "input")
            self._validate_terminal(value.output, "output")
            return value
        component = self._component_of(value)
        if len(component.sides) == 1:
            return component.port(component.sides[0])
        if len(component.sides) == 2 and leaving is not None:
            return component.port(component.sides[1 if leaving else 0])
        raise ValueError(
            f"Cannot infer a component port for component {component.label!r} with "
            f"{len(component.sides)} ports; pass component.port(k) for a "
            "multi-sided component or use cascade() for a directional one."
        )

    def _endpoint_of(
        self, value: Port | SLHComponent | FieldTerminal, direction: TerminalDirection
    ) -> FieldTerminal:
        if isinstance(value, FieldTerminal):
            return value
        component = self._component_of(value)
        return component.input if direction == "input" else component.output

    def _component_of(self, value: Port | SLHComponent) -> SLHComponent:
        return self._component_for_port(value) if isinstance(value, Port) else value

    @staticmethod
    def _cache_value(value: Any) -> Any:
        if value is None:
            return None
        if contains_tracer(value):
            raise ValueError("Traced network values are not cacheable.")
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    (tuple(map(resolve_label, key)), PortNetwork._cache_value(item))
                    for key, item in value.items()
                )
            )
        try:
            array = np.asarray(value)
        except Exception:
            return repr(value)
        return (array.shape, array.dtype.str, array.tobytes())

    @staticmethod
    def _is_concrete_one(value: Any) -> bool:
        if contains_tracer(value):
            return False
        try:
            return bool(np.allclose(np.asarray(value), 1.0))
        except Exception:
            return False

    @staticmethod
    def _is_concrete_zero(value: Any) -> bool:
        if contains_tracer(value):
            return False
        try:
            return bool(np.allclose(np.asarray(value), 0.0))
        except Exception:
            return False


__all__ = ["PortNetwork"]


# Compatibility class names; remove in quchip 0.5.
FieldExposure = NetworkPort
FieldSide = ComponentPort
