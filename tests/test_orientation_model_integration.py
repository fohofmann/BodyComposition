from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from BodyComposition.orientation import OrientationState, assess_orientation
from BodyComposition.orientation.core import apply_model_rotation_to_sitk
from BodyComposition.orientation.ctdeeprot import (
    LPS_YXZ_REFERENCE_CLASS_INDEX,
    CTDeepRotPredictor,
)
from BodyComposition.orientation.rotations import identity_class_index

RUN_VARIABLE = "BODYCOMPOSITION_RUN_CTDEEPROT_TEST"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "tests/fixtures/public_ct/ct_org_volume_0.json"


@pytest.mark.model_integration
@pytest.mark.real_world
def test_ctdeeprot_execution_regression_on_ct_org():
    """Execution regression only; CT-ORG volume 0 is not orientation truth."""

    if os.environ.get(RUN_VARIABLE) != "1":
        pytest.skip(f"set {RUN_VARIABLE}=1 to run the CTDeepRot public-CT test")

    ct_path = os.environ.get("BODYCOMPOSITION_PUBLIC_CT_PATH")
    model_root = os.environ.get("BODYCOMPOSITION_MODEL_ROOT")
    assert ct_path, "BODYCOMPOSITION_PUBLIC_CT_PATH must select the pinned public CT"
    assert model_root, "BODYCOMPOSITION_MODEL_ROOT must select the verified model cache"
    checkpoint_path = (
        Path(model_root)
        / "CTDeepRot"
        / "492114b8f9f3a7f058d4e97c0dd3643fb8d39649"
        / "net2d.pt"
    )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert Path(ct_path).stat().st_size == manifest["integrity"]["bytes"]
    predictor = CTDeepRotPredictor(
        checkpoint_path,
        device="cpu",
    )
    image = sitk.DICOMOrient(sitk.ReadImage(ct_path), "LPS")
    outcome = assess_orientation(image, predictor=predictor)

    assert outcome.result.prediction["raw_winning_class_index"] == 0
    assert (
        outcome.result.prediction["model_reference_class_index"]
        == LPS_YXZ_REFERENCE_CLASS_INDEX
    )
    assert outcome.result.prediction["winning_class_index"] == identity_class_index()
    assert outcome.result.prediction["agreement_count"] >= 23
    assert outcome.result.state is OrientationState.PASS_METADATA_MATCH
    assert not outcome.result.orientation_changed
    assert not outcome.result.manual_review_required

    for injected_class in range(24):
        rotated = apply_model_rotation_to_sitk(
            image,
            injected_class,
            inverse=False,
        )
        prediction = predictor.predict(rotated)
        assert prediction.winning_class_index == injected_class
        assert prediction.agreement_count >= 23

    rotated = apply_model_rotation_to_sitk(image, 8, inverse=False)
    rotated_outcome = assess_orientation(rotated, predictor=predictor)

    assert rotated_outcome.result.prediction["winning_class_index"] == 8
    assert rotated_outcome.result.prediction["agreement_count"] >= 23
    assert rotated_outcome.result.state is OrientationState.MISMATCH_REPAIRED
    assert rotated_outcome.result.orientation_changed
    assert rotated_outcome.result.manual_review_required
    assert rotated_outcome.result.applied_transform["inverse_array_exact"]
    assert np.array_equal(
        sitk.GetArrayFromImage(rotated_outcome.prepared_image),
        sitk.GetArrayFromImage(image),
    )
