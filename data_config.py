from pathlib import Path
import os

DEFAULT_DATA_PATHS = [
    Path(__file__).parent / "FinalData.csv",
    Path(__file__).parent.parent / "kc135" / "kc_135.csv",
]

WUC_LOOKUP_PATHS = [
    Path(__file__).parent / "kc135_wuc_lookup_levels.csv",
    Path(__file__).parent / "kc135_wuc_lookup_dictionary.csv",
]


def resolve_data_path() -> Path:
    override = os.environ.get("WUC_DATA_PATH")
    if override and Path(override).exists():
        return Path(override)
    for p in DEFAULT_DATA_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No maintenance data CSV found. Set WUC_DATA_PATH or place FinalData.csv in the repo."
    )


def resolve_lookup_path() -> Path | None:
    for p in WUC_LOOKUP_PATHS:
        if p.exists():
            return p
    return None


WHEN_DISCOVERED_PHASE = {
    "A": "Pre-flight (ground crew)",
    "B": "In-flight (air crew)",
    "C": "Post-flight (air crew)",
    "D": "Scheduled inspection",
    "E": "Unscheduled inspection",
    "F": "Special inspection",
    "G": "Acceptance inspection",
    "H": "Phase / ISO inspection",
    "J": "Depot / PDM",
    "K": "Functional / test flight",
    "M": "Servicing",
    "Q": "TCTO compliance",
    "3": "Other / undocumented",
}

TYPE_MAINT_PHASE = {
    "A": "On-equipment unscheduled",
    "B": "On-equipment scheduled",
    "C": "Off-equipment unscheduled",
    "D": "Off-equipment scheduled",
    "E": "Look phase",
    "M": "Modification",
    "P": "Periodic / phase",
    "S": "Special inspection",
    "Y": "Deferred / delayed",
}
