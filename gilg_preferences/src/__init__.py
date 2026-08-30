def __getattr__(name: str):
    """Lazy imports to keep generator fast."""
    _type_names = {
        "Message",
        "PreferenceType",
        "PreferencePrompt",
        "MeasurementResponse",
        "BinaryPreferenceMeasurement",
        "TaskScore",
    }
    if name in _type_names:
        from . import types
        return getattr(types, name)

    if name == "MeasurementRecorder":
        from .measurement.elicitation.recorder import MeasurementRecorder
        return MeasurementRecorder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Message",
    "PreferenceType",
    "PreferencePrompt",
    "MeasurementResponse",
    "BinaryPreferenceMeasurement",
    "TaskScore",
    "MeasurementRecorder",
]
