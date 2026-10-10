"""Dressed analysis, fits and the weak-probe VNA of conserving chips through excitation sectors."""
import numpy as np
import pytest

from quchip import RWA, VNA, Capacitive, Chip, DuffingTransmon, PortNetwork, Resonator, fit_a_dress


def _ring(n, *, rate=0.01):
    """n transmons in a ring, a coupling resonator between neighbours and one readout port per transmon."""
    qubits = [DuffingTransmon(freq=5.32 - 0.07 * i, anharmonicity=-0.26, levels=3, label=f"q{i}") for i in range(n)]
    couplers = [Resonator(freq=6.30 + 0.02 * i, levels=2, label=f"c{i}") for i in range(n)]
    readouts = [Resonator(freq=6.56 + 0.05 * i, levels=2, label=f"r{i}") for i in range(n)]
    couplings = []
    for i in range(n):
        couplings += [Capacitive(qubits[i], couplers[i], g=0.03), Capacitive(qubits[(i + 1) % n], couplers[i], g=0.03),
                      Capacitive(qubits[i], readouts[i], g=0.04)]
    network = PortNetwork(label="readout")
    for readout in readouts:
        network.port(readout.label, target=readout, rate=rate)
    return Chip([*qubits, *couplers, *readouts], couplings, port_network=network, frame=5.2, approximation=RWA())


def test_24_mode_ring_is_analysed_from_its_one_excitation_block():
    """Dressed frequencies and weak-probe S of 429,981,696 product states match the hand-built 24-state block."""
    rate = 0.01
    chip = _ring(8, rate=rate)
    labels = [device.label for device in chip.devices]
    block = np.diag([device.freq for device in chip.devices])
    for coupling in chip.couplings:
        a, b = labels.index(coupling.device_a_label), labels.index(coupling.device_b_label)
        block[a, b] = block[b, a] = coupling.g
    energies, vectors = np.linalg.eigh(block)

    frequencies = chip.freq()
    expected = energies[np.argmax(np.abs(vectors) ** 2, axis=1)]
    np.testing.assert_allclose([frequencies[label] for label in labels], expected, rtol=0, atol=1e-12)

    probes = np.array([frequencies[f"r{i}"] for i in range(8)])
    result = VNA(chip).sweep(probes)
    ports = [labels.index(port) for port in result.ports]
    damping = np.diag([rate / 2 if index in ports else 0.0 for index in range(len(labels))])
    responses = [np.linalg.inv(damping + 2j * np.pi * (probe * np.eye(24) - block)) for probe in probes]
    assert result.diagnostics[0]["solver"] == "vacuum_response"
    # On resonance, the response is 2 / rate times larger than S, which sets the round-off.
    np.testing.assert_allclose(np.asarray(result.matrix),
                               [np.eye(8) - rate * response[np.ix_(ports, ports)] for response in responses],
                               rtol=0, atol=1e-10)


def test_fit_limit_applies_to_the_largest_diagonalized_sector():
    """A fit of a conserving pair compares max_hilbert_dim with its largest sector, not its 9 product states."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.4, anharmonicity=-0.22, levels=3, label="q1")
    chip = Chip([q0, q1], [Capacitive(q0, q1, g=0.01, label="q0_q1")])
    constraints = {"q0_q1": {"cross_kerr": None}}

    result = fit_a_dress(chip, constraints=constraints, max_hilbert_dim=3)

    assert result.chip.freq("q0") == pytest.approx(5.0, abs=1e-10)
    assert result.chip.dressed_anharmonicity("q1") == pytest.approx(-0.22, abs=1e-10)
    with pytest.raises(ValueError, match="diagonalizes a matrix of dimension 3, exceeding max_hilbert_dim=2"):
        fit_a_dress(chip, constraints=constraints, max_hilbert_dim=2)
