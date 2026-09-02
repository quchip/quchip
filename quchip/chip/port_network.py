"""Composable, instantaneous SLH field boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quchip.chip.ports import Port
from quchip.utils.jax_utils import contains_tracer, select_array_module
from quchip.utils.labeling import auto_label, resolve_label

if TYPE_CHECKING:
    from quchip.engine.ir import ResolvedSLH


TerminalDirection = Literal["input", "output"]
TerminalKey = tuple[str, str]


@dataclass(frozen=True)
class FieldTerminal:
    """One directional terminal owned by a :class:`PortNetwork`."""

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
    """Minimal public boundary component: scalar ``S`` plus named terminals."""

    label: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    scattering: Any
    _network_token: object = field(repr=False, compare=False)
    _local_ports: tuple[Port | None, ...] = field(default=(), repr=False, compare=False)
    _hidden_pairs: tuple[tuple[str, str, str], ...] = field(
        default=(), repr=False, compare=False
    )

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
        """Return one named input terminal."""
        try:
            index = self.input_names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"No input {name!r} on component {self.label!r}; available: {list(self.input_names)}"
            ) from exc
        return self.inputs[index]

    def output_terminal(self, name: str) -> FieldTerminal:
        """Return one named output terminal."""
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
class _Exposure:
    label: str
    input_key: TerminalKey
    output_key: TerminalKey
    delay: Any = 0.0
    hidden: bool = False


@dataclass
class _AffineField:
    scattering: list[Any]
    coupling: dict[str, Any]


class PortNetwork:
    """Ports and an acyclic, instantaneous scalar-S field-routing graph."""

    _type_prefix = "port_network"

    def __init__(
        self,
        *,
        scattering: Any = None,
        label: str | None = None,
    ) -> None:
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
        self._component_kinds: dict[str, str] = {}
        self._component_parameters: dict[str, dict[str, Any]] = {}
        self._connections: dict[TerminalKey, TerminalKey] = {}
        self._used_outputs: dict[TerminalKey, TerminalKey] = {}
        self._exposures: list[_Exposure] = []

    @classmethod
    def from_ports(
        cls,
        ports: Sequence[Port],
        *,
        scattering: Any = None,
        label: str | None = None,
    ) -> "PortNetwork":
        """Build an identity-exposed network around existing ports."""
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
    def exposures(self) -> tuple[str, ...]:
        """Return explicitly authored external exposure labels."""
        return tuple(exposure.label for exposure in self._exposures if not exposure.hidden)

    @property
    def S(self) -> Any:
        """Return the composed instantaneous scalar scattering matrix."""
        return self._compile()[1]

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return bindable scalar network parameters visible in authored scattering."""
        values: dict[str, Any] = {}
        if isinstance(self._authored_scattering, Mapping):
            for (output, input_), value in self._authored_scattering.items():
                values[f"scattering.{resolve_label(output)}.{resolve_label(input_)}"] = value
        for exposure in self._exposures:
            if not exposure.hidden:
                values[f"exposure.{exposure.label}.delay"] = exposure.delay
        for label, parameters in self._component_parameters.items():
            for name, value in parameters.items():
                values[f"component.{label}.{name}"] = value
        return MappingProxyType(values)

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Set one network-owned scalar on an isolated structural copy."""
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
        if len(parts) == 3 and parts[0] == "exposure" and parts[2] == "delay":
            for index, exposure in enumerate(self._exposures):
                if exposure.label == parts[1]:
                    self._exposures[index] = replace(exposure, delay=value)
                    return
        if len(parts) == 3 and parts[0] == "component":
            parameters = self._component_parameters.get(parts[1])
            if parameters is not None and parts[2] in parameters:
                parameters[parts[2]] = value
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
        """Create and return one quantum coupling port owned by this network."""
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
        """Add a zero-coupling scalar scattering component."""
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
        )
        return self._add_component(component)

    def through(self, label: str) -> SLHComponent:
        """Add a one-channel identity through component."""
        return self.component(label, scattering=[[1.0]], terminals=("signal",))

    def phase_shift(self, label: str, *, phase: Any) -> SLHComponent:
        """Add a one-channel phase shift."""
        xp = select_array_module(contains_tracer(phase))
        component = self.component(
            label,
            scattering=xp.asarray([[xp.exp(1j * xp.asarray(phase))]]),
            terminals=("signal",),
        )
        self._component_kinds[label] = "phase_shift"
        self._component_parameters[label] = {"phase": phase}
        return component

    def beam_splitter(self, label: str, *, eta: Any = 0.5) -> SLHComponent:
        """Add a reciprocal two-channel splitter parameterized by power transmission."""
        matrix = self._transmission_matrix(eta)
        component = self.component(label, scattering=matrix, terminals=("left", "right"))
        self._component_kinds[label] = "beam_splitter"
        self._component_parameters[label] = {"eta": eta}
        return component

    def hybrid90(self, label: str) -> SLHComponent:
        """Add an ideal reciprocal 90-degree hybrid."""
        return self.component(
            label,
            scattering=np.asarray([[1.0, 1j], [1j, 1.0]], dtype=complex) / np.sqrt(2.0),
            terminals=("left", "right"),
        )

    def permutation(self, label: str, *, order: Sequence[int]) -> SLHComponent:
        """Add an output permutation; ``order[row]`` selects the input column."""
        order = tuple(order)
        if sorted(order) != list(range(len(order))):
            raise ValueError("Permutation order must contain each input index exactly once.")
        matrix = np.zeros((len(order), len(order)), dtype=complex)
        matrix[np.arange(len(order)), order] = 1.0
        return self.component(
            label,
            scattering=matrix,
            terminals=tuple(str(index) for index in range(len(order))),
        )

    def attenuator(self, label: str, *, eta: Any) -> SLHComponent:
        """Add a lossless dilation of power transmission ``eta`` and hidden vacuum."""
        concrete = None
        if not contains_tracer(eta):
            try:
                concrete = float(np.asarray(eta))
            except (TypeError, ValueError):
                concrete = None
        if concrete is not None and not 0.0 <= concrete <= 1.0:
            raise ValueError(f"Attenuator eta must lie in [0, 1], got {eta}.")
        component = SLHComponent(
            label=label,
            input_names=("signal", "vacuum"),
            output_names=("signal", "vacuum"),
            scattering=self._transmission_matrix(eta),
            _network_token=self._token,
            _local_ports=(None, None),
            _hidden_pairs=(("vacuum", "vacuum", f"hidden.{label}.vacuum"),),
        )
        component = self._add_component(component)
        self._component_kinds[label] = "attenuator"
        self._component_parameters[label] = {"eta": eta}
        return component

    def connect(self, output: FieldTerminal, input: FieldTerminal) -> None:
        """Connect one component output to one component input."""
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

    def cascade(self, first: Port | SLHComponent, second: Port | SLHComponent) -> None:
        """Connect the sole or signal output of ``first`` to the input of ``second``."""
        self.connect(self._output_of(first), self._input_of(second))

    def expose(
        self,
        label: str,
        *,
        input: FieldTerminal,
        output: FieldTerminal,
        delay: Any = 0.0,
    ) -> None:
        """Name one external input/output pair and its reciprocal reference delay."""
        self._validate_terminal(input, "input")
        self._validate_terminal(output, "output")
        self._reject_hidden_terminal(input)
        self._reject_hidden_terminal(output)
        if label in {exposure.label for exposure in self._exposures}:
            raise ValueError(f"Duplicate PortNetwork exposure label {label!r}.")
        if input.key in self._connections or output.key in self._used_outputs:
            raise ValueError("Connected terminals cannot also be exposed.")
        if self._terminal_is_exposed(input) or self._terminal_is_exposed(output):
            raise ValueError("A terminal cannot belong to more than one exposure.")
        concrete = None
        if not contains_tracer(delay):
            try:
                concrete = float(np.asarray(delay))
            except (TypeError, ValueError):
                concrete = None
        if concrete is not None and concrete < 0:
            raise ValueError("Exposure delay must be non-negative.")
        self._exposures.append(_Exposure(label, input.key, output.key, delay))

    def validate_for(self, chip: Any) -> None:
        """Validate every quantum port target against ``chip``."""
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
                    self._cache_value(component.scattering),
                    tuple(
                        (name, self._cache_value(value))
                        for name, value in self._component_parameters.get(component.label, {}).items()
                    ),
                )
                for component in self.components
            ),
            tuple(self._connections.items()),
            tuple(
                (item.label, item.input_key, item.output_key, self._cache_value(item.delay))
                for item in self._exposures
            ),
            self._cache_value(self._authored_scattering),
        )

    def resolve(self, base: "ResolvedSLH") -> "ResolvedSLH":
        """Compose this boundary onto port channels in an input-free SLH value."""
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

        exposures, scattering, coupling_maps, generated_pairs = self._compile()
        operators = {label: port_channels[label].coupling for label in expected}
        resolved_couplings = [
            self._materialize_coupling(mapping, operators, key=exposure.label)
            for exposure, mapping in zip(exposures, coupling_maps, strict=True)
        ]

        channels: list[SLHChannel] = []
        for exposure, mapping, coupling in zip(
            exposures, coupling_maps, resolved_couplings, strict=True
        ):
            if (
                exposure.label in port_channels
                and set(mapping) == {exposure.label}
                and self._is_concrete_one(mapping[exposure.label])
            ):
                channels.append(
                    replace(
                        port_channels[exposure.label],
                        key=exposure.label,
                        accessibility="hidden" if exposure.hidden else "exposed",
                        reference_delay=exposure.delay,
                    )
                )
                continue
            source_channel = (
                port_channels[next(iter(mapping))]
                if len(mapping) == 1
                else next(iter(port_channels.values()))
            )
            template = CollapseTerm(
                operator=coupling,
                rate=1.0,
                source=exposure.label,
                channel="network",
                frame_frequency=source_channel.collapse.frame_frequency,
            )
            channels.append(
                SLHChannel(
                    key=exposure.label,
                    accessibility="hidden" if exposure.hidden else "exposed",
                    collapse=template,
                    coupling_operator=coupling,
                    reference_delay=exposure.delay,
                )
            )

        generated = self._materialize_generated_hamiltonian(generated_pairs, operators)
        static_terms = base.H.static_terms
        if generated is not None:
            static_terms = (*static_terms, StaticTerm(operator=generated, origin="network"))

        hidden = base.hidden_channels
        full_size = len(channels) + len(hidden)
        xp = select_array_module(contains_tracer(scattering))
        full_scattering = xp.eye(full_size, dtype=complex)
        if channels:
            if hasattr(full_scattering, "at"):
                full_scattering = full_scattering.at[: len(channels), : len(channels)].set(scattering)
            else:
                full_scattering[: len(channels), : len(channels)] = scattering
        return ResolvedSLH(
            scattering=full_scattering,
            hamiltonian=HamiltonianProgram(
                static_terms=static_terms,
                dynamic_terms=base.H.dynamic_terms,
            ),
            channels=tuple((*channels, *hidden)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize a static network graph and its quantum ports."""
        return {
            "label": self.label,
            "ports": [port.to_dict() for port in self._ports],
            "scattering": self._serialize_boundary(self._authored_scattering),
            "components": [
                {
                    "label": component.label,
                    "input_names": list(component.input_names),
                    "output_names": list(component.output_names),
                    "scattering": self._serialize_matrix(component.scattering),
                    "hidden_pairs": [list(item) for item in component._hidden_pairs],
                    "kind": self._component_kinds.get(component.label, "scattering"),
                    "parameters": {
                        name: self._serialize_scalar(value)
                        for name, value in self._component_parameters.get(component.label, {}).items()
                    },
                }
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
                    "input": list(exposure.input_key),
                    "output": list(exposure.output_key),
                    "delay": self._serialize_scalar(exposure.delay),
                }
                for exposure in self._exposures
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PortNetwork":
        """Reconstruct a static network produced by :meth:`to_dict`."""
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
            network._add_component(
                SLHComponent(
                    label=payload["label"],
                    input_names=tuple(payload["input_names"]),
                    output_names=tuple(payload["output_names"]),
                    scattering=cls._deserialize_matrix(payload["scattering"]),
                    _network_token=network._token,
                    _local_ports=tuple(None for _ in payload["output_names"]),
                    _hidden_pairs=tuple(tuple(item) for item in payload.get("hidden_pairs", [])),
                )
            )
            network._component_kinds[payload["label"]] = payload.get("kind", "scattering")
            network._component_parameters[payload["label"]] = {
                name: cls._deserialize_scalar(value)
                for name, value in payload.get("parameters", {}).items()
            }
        for payload in data.get("connections", []):
            input_key = tuple(payload["input"])
            output_key = tuple(payload["output"])
            network._connections[input_key] = output_key
            network._used_outputs[output_key] = input_key
        for payload in data.get("exposures", []):
            network._exposures.append(
                _Exposure(
                    payload["label"],
                    tuple(payload["input"]),
                    tuple(payload["output"]),
                    cls._deserialize_scalar(payload.get("delay", 0.0)),
                )
            )
        return network

    def copy(self) -> "PortNetwork":
        """Return an independent structural copy with label-based port targets."""
        copied = PortNetwork(
            scattering=self._copy_value(self._authored_scattering),
            label=self.label,
        )
        for component in self.components:
            if any(port is not None for port in component._local_ports):
                assert len(component._local_ports) == 1
                port = component._local_ports[0]
                assert port is not None
                copied._add_port(port.copy())
            else:
                copied._add_component(
                    SLHComponent(
                        label=component.label,
                        input_names=component.input_names,
                        output_names=component.output_names,
                        scattering=self._copy_value(component.scattering),
                        _network_token=copied._token,
                        _local_ports=component._local_ports,
                        _hidden_pairs=component._hidden_pairs,
                    )
                )
        copied._connections = dict(self._connections)
        copied._used_outputs = dict(self._used_outputs)
        copied._exposures = list(self._exposures)
        copied._component_kinds = dict(self._component_kinds)
        copied._component_parameters = {
            label: dict(parameters)
            for label, parameters in self._component_parameters.items()
        }
        return copied

    def physics_notes(self) -> list[str]:
        """Return the boundary convention and approximation scope."""
        return [
            "This instantaneous Markovian boundary uses b_out = S b_in + L; "
            "scalar S is unitary after explicit vacuum loss dilation.",
            "Exposure delays move reciprocal external reference planes only and do not enter S.",
        ]

    @staticmethod
    def _serialize_matrix(value: Any) -> dict[str, Any]:
        if contains_tracer(value):
            raise TypeError("Traced PortNetwork scattering cannot be serialized.")
        matrix = np.asarray(value, dtype=complex)
        return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}

    @staticmethod
    def _deserialize_matrix(value: Mapping[str, Any]) -> np.ndarray:
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
    def _deserialize_scalar(value: Any) -> complex | float:
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
            _local_ports=(port,),
        )
        self._add_component(component)
        port._bind_network(self, component)
        self._ports.append(port)

    def _add_component(self, component: SLHComponent) -> SLHComponent:
        if component.label in self._components:
            raise ValueError(f"Duplicate PortNetwork component label {component.label!r}.")
        self._components[component.label] = component
        return component

    def _effective_exposures(self) -> tuple[_Exposure, ...]:
        exposed = list(self._exposures)
        used = {
            key
            for exposure in self._exposures
            for key in (exposure.input_key, exposure.output_key)
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
                exposed.append(_Exposure(component.label, input_key, output_key))
        hidden: list[_Exposure] = []
        for component in self.components:
            for input_name, output_name, label in component._hidden_pairs:
                hidden.append(
                    _Exposure(
                        label,
                        (component.label, input_name),
                        (component.label, output_name),
                        hidden=True,
                    )
                )
        return tuple((*exposed, *hidden))

    def _compile(
        self,
    ) -> tuple[tuple[_Exposure, ...], Any, list[dict[str, Any]], list[tuple[str, str, Any]]]:
        exposures = self._effective_exposures()
        covered_inputs = {exposure.input_key for exposure in exposures}
        covered_outputs = {exposure.output_key for exposure in exposures}
        all_inputs = {
            (component.label, name) for component in self.components for name in component.input_names
        }
        all_outputs = {
            (component.label, name) for component in self.components for name in component.output_names
        }
        free_inputs = all_inputs - set(self._connections) - covered_inputs
        free_outputs = all_outputs - set(self._used_outputs) - covered_outputs
        if free_inputs or free_outputs:
            raise ValueError(
                "PortNetwork has free terminals; connect or expose them explicitly: "
                f"inputs={sorted(free_inputs)}, outputs={sorted(free_outputs)}."
            )
        if len(covered_inputs) != len(exposures) or len(covered_outputs) != len(exposures):
            raise ValueError("PortNetwork exposures must use distinct input and output terminals.")

        size = len(exposures)
        input_fields: dict[TerminalKey, _AffineField] = {}
        output_fields: dict[TerminalKey, _AffineField] = {}
        for column, exposure in enumerate(exposures):
            basis = [0.0] * size
            basis[column] = 1.0
            input_fields[exposure.input_key] = _AffineField(basis, {})

        pending = list(self._components)
        generated_pairs: list[tuple[str, str, Any]] = []
        while pending:
            progressed = False
            for label in tuple(pending):
                component = self._components[label]
                incoming: list[_AffineField] = []
                ready = True
                for name in component.input_names:
                    key = (label, name)
                    if key in input_fields:
                        incoming.append(input_fields[key])
                        continue
                    upstream_output = self._connections.get(key)
                    if upstream_output is None or upstream_output not in output_fields:
                        ready = False
                        break
                    incoming.append(output_fields[upstream_output])
                if not ready:
                    continue
                matrix = self._component_matrix(component)
                for row, output_name in enumerate(component.output_names):
                    scattering = [
                        sum(matrix[row, column] * incoming[column].scattering[index] for column in range(len(incoming)))
                        for index in range(size)
                    ]
                    upstream: dict[str, Any] = {}
                    for column, incoming_field in enumerate(incoming):
                        for coupling_source, coefficient in incoming_field.coupling.items():
                            upstream[coupling_source] = (
                                upstream.get(coupling_source, 0.0)
                                + matrix[row, column] * coefficient
                            )
                    local_port = component._local_ports[row]
                    coupling = dict(upstream)
                    if local_port is not None:
                        local_label = local_port.label
                        for coupling_source, coefficient in upstream.items():
                            generated_pairs.append((local_label, coupling_source, coefficient))
                        coupling[local_label] = coupling.get(local_label, 0.0) + 1.0
                    output_fields[(label, output_name)] = _AffineField(scattering, coupling)
                pending.remove(label)
                progressed = True
            if not progressed:
                raise ValueError("PortNetwork contains instantaneous feedback or a connection cycle.")

        rows = [output_fields[exposure.output_key] for exposure in exposures]
        scattering_rows = [row.scattering for row in rows]
        xp = select_array_module(contains_tracer(scattering_rows))
        scattering = xp.asarray(scattering_rows, dtype=complex)
        coupling_maps = [row.coupling for row in rows]

        external_count = sum(not exposure.hidden for exposure in exposures)
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

        self._validate_unitary(scattering)
        return exposures, scattering, coupling_maps, generated_pairs

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
    def _materialize_coupling(
        mapping: dict[str, Any],
        operators: dict[str, Any],
        *,
        key: str,
    ) -> Any:
        from quchip.engine.ir import CanonicalOperator

        if not operators:
            raise RuntimeError("A PortNetwork cannot resolve without quantum coupling ports.")
        first = next(iter(operators.values()))
        prefer_jax = contains_tracer((mapping, tuple(operator.values for operator in operators.values())))
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
        pairs: list[tuple[str, str, Any]],
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
        kind = self._component_kinds.get(component.label)
        parameters = self._component_parameters.get(component.label, {})
        if kind == "phase_shift":
            phase = parameters["phase"]
            xp = select_array_module(contains_tracer(phase))
            matrix = xp.asarray([[xp.exp(1j * xp.asarray(phase))]])
        elif kind in {"beam_splitter", "attenuator"}:
            matrix = self._transmission_matrix(parameters["eta"])
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
        xp = select_array_module(contains_tracer(eta))
        transmission = xp.sqrt(xp.asarray(eta))
        loss = xp.sqrt(1.0 - xp.asarray(eta))
        return xp.asarray([[transmission, loss], [-loss, transmission]], dtype=complex)

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
            terminal.key in (exposure.input_key, exposure.output_key)
            for exposure in self._exposures
        )

    def _component_for_port(self, port: Port) -> SLHComponent:
        for component in self.components:
            if any(candidate is port for candidate in component._local_ports):
                return component
        raise ValueError(f"Port {port.label!r} is not owned by this PortNetwork.")

    def _input_of(self, value: Port | SLHComponent) -> FieldTerminal:
        return self._component_for_port(value).input if isinstance(value, Port) else value.input

    def _output_of(self, value: Port | SLHComponent) -> FieldTerminal:
        return self._component_for_port(value).output if isinstance(value, Port) else value.output

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
    def _copy_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    @staticmethod
    def _is_concrete_one(value: Any) -> bool:
        if contains_tracer(value):
            return False
        try:
            return bool(np.allclose(np.asarray(value), 1.0))
        except Exception:
            return False


__all__ = ["FieldTerminal", "PortNetwork", "SLHComponent"]
