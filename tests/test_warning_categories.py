"""Warning categories let users filter quchip warnings by group."""

import warnings

import pytest

from quchip import (AccuracyWarning, AutoChoiceWarning, ChargeBasisTransmon, ChargeDrive, Chip, DuffingTransmon,
                    QuantumSequence, QuchipWarning, Square)

pytestmark = pytest.mark.unit


def _emitted(*filters):
    """Return the categories of one automatic-choice and one accuracy warning under ``filters``."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    chip = Chip([q], frame="rotating")
    chip.wire(ChargeDrive(q, label="xy"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for category in filters:
            warnings.filterwarnings("ignore", category=category)
        QuantumSequence(chip).charge(q, envelope=Square(duration=10.0, amplitude=0.02))
        ChargeBasisTransmon.from_frequency(freq=2.0, anharmonicity=-0.5)
    return [item.category for item in caught]


def test_each_filter_silences_its_group_and_keeps_the_other():
    assert issubclass(AutoChoiceWarning, QuchipWarning) and issubclass(AccuracyWarning, QuchipWarning)
    assert issubclass(QuchipWarning, UserWarning)
    assert _emitted() == [AutoChoiceWarning, AccuracyWarning]
    assert _emitted(AutoChoiceWarning) == [AccuracyWarning]
    assert _emitted(AccuracyWarning) == [AutoChoiceWarning]
    assert _emitted(QuchipWarning) == []
