"""Optional final case-report action."""

from __future__ import annotations

from pathlib import Path

from BodyComposition.pipeline import PipelineAction
from BodyComposition.reporting.contracts import CaseReportInput, ReportingSettings
from BodyComposition.reporting.service import render_case_report

CASE_REPORT_PDF = "reports/case_report.pdf"
CASE_REPORT_MANIFEST = "reports/report_manifest.json"
TISSUE_LABEL_MASK = "masks/tissue_labels.nii.gz"


class RenderCaseReport(PipelineAction):
    """Render one derived case page after authoritative measurement export."""

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.settings = ReportingSettings.from_mapping(self.config)
        if not self.settings.enabled:
            raise ValueError("RenderCaseReport cannot be constructed while reporting is disabled.")
        self.io_inputs = [
            "tmp/prepared_image",
            "tmp/orientation_result",
            "tmp/vertebral_result",
            "tmp/measurement_bundle",
            TISSUE_LABEL_MASK,
        ]
        self.io_outputs = [CASE_REPORT_PDF, CASE_REPORT_MANIFEST, "tmp/report_result"]
        self.io_persisted_outputs = [CASE_REPORT_PDF, CASE_REPORT_MANIFEST]
        self.io_reset_outputs = [CASE_REPORT_PDF, CASE_REPORT_MANIFEST]

    def __call__(self, memory):
        super().__call__(memory)
        case_id = str(memory["id"])
        report_input = CaseReportInput(
            case_id=case_id,
            prepared_image=memory["tmp/prepared_image"].prepared_image,
            vertebral_result=memory["tmp/vertebral_result"],
            measurement_bundle=memory["tmp/measurement_bundle"],
            tissue_labels_zyx=memory[TISSUE_LABEL_MASK].data,
            orientation=memory["tmp/orientation_result"],
            technical_metadata=dict(memory.get("tmp/report_metadata", {})),
        )
        output_directory = Path(memory["workspace"]) / "reports"
        result = render_case_report(report_input, output_directory, self.settings)
        memory[CASE_REPORT_PDF] = result.pdf_path
        memory[CASE_REPORT_MANIFEST] = result.manifest_path
        memory["tmp/report_result"] = result
