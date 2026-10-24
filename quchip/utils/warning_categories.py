"""Warning categories, so that users can filter quchip warnings by group."""


class QuchipWarning(UserWarning):
    """Base category of every quchip warning except deprecations.

    Renamed or removed APIs keep :class:`DeprecationWarning`. Filter this
    category to silence every other quchip warning.
    """


class AutoChoiceWarning(QuchipWarning):
    """A notice that quchip chose a setting for you, such as a frame, a carrier or a cutoff.

    The choice is safe to keep. Pass the setting explicitly to make the
    notice go away.
    """


class AccuracyWarning(QuchipWarning):
    """Advice that a result can be inaccurate, such as a truncation, an approximation or a weak fit."""
