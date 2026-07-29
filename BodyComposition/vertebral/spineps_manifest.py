"""Pinned software, model assets, and native labels for SPINEPS/VERIDAH CT."""

from __future__ import annotations

from dataclasses import dataclass

SPINEPS_BACKEND_ID = "spineps_veridah_ct_v1"
SPINEPS_VERSION = "2.0.0"
SPINEPS_SOURCE_COMMIT = "ad622b87d9e4b81fb6a88df2a8050bd40f7046d5"
SPINEPS_WHEEL_SHA256 = "5a5feec1268b71ccf5d2c4400cd7f4ca68d7673bdb16814956e238de18c5b40b"
SPINEPS_SOURCE_LICENSE = "Apache-2.0"

TPTBOX_COMPAT_VERSION = "0.7.5"
TPTBOX_SOURCE_LICENSE = "Apache-2.0"

VIBESEG_SOURCE_REPOSITORY = "https://github.com/robert-graf/VIBESegmentator"
VIBESEG_SOURCE_LICENSE = "Apache-2.0"
VIBESEG_WEIGHT_LICENSE_STATUS = "upstream_sync_only_not_redistributed"
MODEL_WEIGHT_REDISTRIBUTION_MODE = "user_model_sync"
VIBESEG_CROP_DATASET_ID = 100
VIBESEG_CROP_RELEASE = "v1.0.0"
MODEL_BUNDLE_VERSION = "spineps-veridah-ct-v1"
# Canonical SHA-256 digests of the complete expanded file inventories produced
# by the exact size/SHA-pinned upstream archives below. These package-owned
# pins authenticate installed caches without retaining duplicate source ZIPs.
VIBESEG_CROP_INSTALLED_INVENTORY_SHA256 = (
    "755b78daacf25d9b6496cb1dec424b130ed8acff62b41d0980300e0a50b4fcc2"
)

VERTEBRA_CORPUS_SEMANTIC_LABEL = 49
SACRUM_BODY_SEMANTIC_LABEL = 73

SPINEPS_NATIVE_LABELS: dict[int, str] = {
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
    5: "C5",
    6: "C6",
    7: "C7",
    8: "T1",
    9: "T2",
    10: "T3",
    11: "T4",
    12: "T5",
    13: "T6",
    14: "T7",
    15: "T8",
    16: "T9",
    17: "T10",
    18: "T11",
    19: "T12",
    20: "L1",
    21: "L2",
    22: "L3",
    23: "L4",
    24: "L5",
    25: "L6",
    26: "SACRUM",
    28: "T13",
}

SPINEPS_CRANIO_CAUDAL_ORDER = (
    *range(1, 20),
    28,
    *range(20, 27),
)

# SPINEPS 2.0.0 stores derived IVD and endplate instances in the vertebral
# output as label+100 and label+200. Importing the upstream constants also
# initializes its model runtime, so these pinned ranges are checked against
# the upstream package by an equivalence test.
SPINEPS_AUXILIARY_INSTANCE_LABELS = frozenset(
    (*range(100, 134), *range(200, 234))
)


@dataclass(frozen=True)
class ModelAssetSpec:
    """One exact upstream archive installed into the BodyComposition bundle."""

    model_id: str
    phase: str
    release: str
    asset_name: str
    url: str
    bytes: int
    sha256: str
    installed_inventory_sha256: str
    install_dir: str

    def __post_init__(self) -> None:
        if self.phase not in {"semantic", "instance", "labeling"}:
            raise ValueError(f"Unknown SPINEPS model phase: {self.phase}.")
        if self.bytes <= 0:
            raise ValueError("Model archive size must be positive.")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("Model archive SHA-256 must be lowercase hexadecimal.")
        if len(self.installed_inventory_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.installed_inventory_sha256
        ):
            raise ValueError("Installed model inventory SHA-256 must be lowercase hexadecimal.")
        if not self.url.startswith("https://github.com/Hendrik-code/spineps/releases/download/"):
            raise ValueError("SPINEPS model assets must use an exact upstream release URL.")


@dataclass(frozen=True)
class ReleaseAssetPin:
    """One immutable release asset required by an upstream support model."""

    asset_name: str
    url: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.bytes <= 0:
            raise ValueError("Release asset size must be positive.")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("Release asset SHA-256 must be lowercase hexadecimal.")
        if not self.url.startswith(
            "https://github.com/robert-graf/VIBESegmentator/releases/download/"
        ):
            raise ValueError("VibeSeg assets must use an exact upstream release URL.")


SPINEPS_MODEL_ASSETS = (
    ModelAssetSpec(
        model_id="ct",
        phase="semantic",
        release="v1.4.2",
        asset_name="ct.zip",
        url="https://github.com/Hendrik-code/spineps/releases/download/v1.4.2/ct.zip",
        bytes=765_437_777,
        sha256="744a18a2a25955b2a45deff15b3dda2425fd80b715f8e97500151405abaa495e",
        installed_inventory_sha256=(
            "349936738121fb0ab396e2a910528d8d7110a5cf043b87dff5d71a00e6116a50"
        ),
        install_dir="semantic_ct",
    ),
    ModelAssetSpec(
        model_id="ct_instance",
        phase="instance",
        release="v1.4.2",
        asset_name="CT_instance.zip",
        url="https://github.com/Hendrik-code/spineps/releases/download/v1.4.2/CT_instance.zip",
        bytes=5_160_620,
        sha256="ea943c44afb46782dfe4cd66fc017837c191e555087fa9490fadd8fd8c413cf7",
        installed_inventory_sha256=(
            "d8b58de9d436f1d786279a7d17af9bd66aa1acc8bebcd584db72b7f4bc4b2b56"
        ),
        install_dir="instance_ct",
    ),
    ModelAssetSpec(
        model_id="ct_labeling",
        phase="labeling",
        release="v1.4.0",
        asset_name="ct_labeling.zip",
        url="https://github.com/Hendrik-code/spineps/releases/download/v1.4.0/ct_labeling.zip",
        bytes=87_702_976,
        sha256="a3b583d9a1dca19a92b7ccb466d08ba7464fa90752ef7e2be53817d4b43496f5",
        installed_inventory_sha256=(
            "3f3fe9a26820447fb01a38dfe862e218867a1cb37f2dded032719c37c0b6d4a7"
        ),
        install_dir="labeling_ct",
    ),
)

SPINEPS_MODEL_ASSET_BY_ID = {asset.model_id: asset for asset in SPINEPS_MODEL_ASSETS}

# SPINEPS 2.0.0 requires TPTBox VibeSeg Dataset100 for CT cropping. The
# upstream manifest splits the approximately 2.3 GB model across three
# archives; each archive is pinned because inference-time downloads are
# disabled.
VIBESEG_CROP_ASSETS = (
    ReleaseAssetPin(
        asset_name="100.zip",
        url="https://github.com/robert-graf/VIBESegmentator/releases/download/v1.0.0/100.zip",
        bytes=767_635_241,
        sha256="97483c90952bd988c58e8a040a59e685183a259d903c42e25f1b78224082f4fe",
    ),
    ReleaseAssetPin(
        asset_name="100_1.zip",
        url="https://github.com/robert-graf/VIBESegmentator/releases/download/v1.0.0/100_1.zip",
        bytes=767_634_617,
        sha256="529e033aeb9fb1fa8b6eca9399686cbb73820dfb5a6390bf62d313be1c58c219",
    ),
    ReleaseAssetPin(
        asset_name="100_2.zip",
        url="https://github.com/robert-graf/VIBESegmentator/releases/download/v1.0.0/100_2.zip",
        bytes=767_276_323,
        sha256="f1a72029371d99823ed2dfafaa1c97a7b1ca6e20a7e78aece100025af8143102",
    ),
)
