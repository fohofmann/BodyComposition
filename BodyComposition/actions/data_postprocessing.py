from __future__ import annotations

import logging
from pathlib import Path
from time import time
from typing import Callable

import numpy as np
import pandas as pd

from BodyComposition.measurements import weighted_mean_hu
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import assert_same_physical_domain


MEASUREMENT_PREFIXES = ("Vx_", "CSA_", "HU_", "CIR_")


def filter_measurement_columns(df: pd.DataFrame, pattern: str | None) -> pd.DataFrame:
    if pattern is None:
        return df
    measurement_columns = [
        column for column in df.columns if column.startswith(MEASUREMENT_PREFIXES)
    ]
    retained_measurements = df[measurement_columns].filter(regex=pattern).columns.tolist()
    other_columns = [column for column in df.columns if column not in measurement_columns]
    return df[other_columns + retained_measurements]


class DataCombine(PipelineAction):
    """Combine aligned vertebral and tissue measurements into one slice table."""

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.io_inputs = [
            "tmp/tissue_values",
            "tmp/tissue_geometry",
            "tmp/vertebrae_values",
            "tmp/vertebrae_geometry",
        ]
        self.io_outputs = ["tmp/bodycomposition"]

    def __call__(self, memory):
        super().__call__(memory)
        time_start = time()

        tissue_geometry = memory["tmp/tissue_geometry"]
        vertebrae_geometry = memory["tmp/vertebrae_geometry"]
        assert_same_physical_domain(
            tissue_geometry,
            vertebrae_geometry,
            reference_name="tissue mask",
            candidate_name="vertebral mask",
        )

        tissue_df = memory["tmp/tissue_values"]
        vertebrae_df = memory["tmp/vertebrae_values"]
        if not isinstance(tissue_df, pd.DataFrame) or not isinstance(vertebrae_df, pd.DataFrame):
            raise TypeError("Tissue and vertebral measurements must be pandas DataFrames.")
        if len(tissue_df) != len(vertebrae_df):
            raise ValueError(
                "Tissue and vertebral tables must have one row per prepared CT slice: "
                f"{len(tissue_df)} != {len(vertebrae_df)}."
            )
        if "Slice" not in vertebrae_df:
            raise ValueError("Vertebral measurements are missing the required Slice column.")

        results_df = pd.concat(
            [vertebrae_df.reset_index(drop=True), tissue_df.reset_index(drop=True)],
            axis=1,
        )
        results_df = results_df.sort_values("Slice", kind="stable").reset_index(drop=True)
        memory["tmp/bodycomposition"] = results_df
        logging.info(
            " output: memory:tmp/bodycomposition, shape %s (%.2fs)",
            results_df.shape,
            time() - time_start,
        )


class L3MeanCSA(PipelineAction):
    """Create a schema-valid L3 summary, including an explicit missing-level status."""

    def __init__(
        self,
        pipeline,
        input_df: str = "tmp/bodycomposition",
        output_df: str = "res/L3MeanCSA",
    ):
        super().__init__(pipeline)
        self.input_df_name = input_df
        self.output_df_name = output_df
        self.csa_columns = [f"CSA_{name}" for name in pipeline.config["LBL_TISSUE"].values()]
        self.io_inputs = [input_df]
        self.io_outputs = [output_df]

    def __call__(self, memory):
        super().__call__(memory)
        input_df = memory[self.input_df_name]
        if not isinstance(input_df, pd.DataFrame):
            raise TypeError("L3MeanCSA input must be a pandas DataFrame.")

        row = {"status": "not_available", "reason": "missing_L3"}
        row.update({column: np.nan for column in self.csa_columns})
        if "Level" in input_df:
            l3_rows = input_df[input_df["Level"].eq("L3")]
            if not l3_rows.empty:
                available_columns = [column for column in self.csa_columns if column in l3_rows]
                row.update(l3_rows[available_columns].mean(numeric_only=True).to_dict())
                row["status"] = "ok"
                row["reason"] = None

        memory[self.output_df_name] = pd.DataFrame([row])


class DataSubset(PipelineAction):
    """Subset a slice table by vertebral reference and level."""

    def __init__(
        self,
        pipeline,
        ref: str,
        level,
        input_df: str = "tmp/bodycomposition",
        output_df: str = "tmp/bodycomposition",
    ):
        super().__init__(pipeline)
        if ref not in {"Level", "Center", "Centroid", "Tag"}:
            raise ValueError(f"Unsupported vertebral reference: {ref}.")
        self.ref = ref
        self.level = level
        self.input_df_name = input_df
        self.output_df_name = output_df
        self.io_inputs = [input_df]
        self.io_outputs = [output_df]

    def __call__(self, memory):
        super().__call__(memory)
        input_df = memory[self.input_df_name]
        if not isinstance(input_df, pd.DataFrame):
            raise TypeError("DataSubset input must be a pandas DataFrame.")
        if self.ref not in input_df:
            raise ValueError(f"Input table is missing the {self.ref} column.")

        reference = input_df[self.ref].astype("string")
        if self.level == "ALL":
            selector = reference.notna()
        elif self.level == "L":
            selector = reference.str.startswith("L", na=False)
        else:
            levels = [self.level] if isinstance(self.level, str) else list(self.level)
            selector = reference.isin(levels)
        memory[self.output_df_name] = input_df.loc[selector].copy().reset_index(drop=True)


class DataAggregate(PipelineAction):
    """Aggregate numeric measurements, optionally grouped by an anatomical reference."""

    def __init__(
        self,
        pipeline,
        method: str | None = None,
        ref: str | None = None,
        tag_mapping: dict | None = None,
        input_df: str = "tmp/bodycomposition",
        output_df: str = "tmp/bodycomposition",
        filter_columns: str | None = None,
    ):
        super().__init__(pipeline)
        if method not in {None, "mean", "median", "sum"}:
            raise ValueError("Aggregation method must be mean, median, sum, or None.")
        self.method = method
        self.ref = ref
        self.tag_mapping = tag_mapping
        self.input_df_name = input_df
        self.output_df_name = output_df
        self.filter_columns = filter_columns
        self.io_inputs = [input_df]
        self.io_outputs = [output_df]

    def __call__(self, memory):
        super().__call__(memory)
        input_df = memory[self.input_df_name]
        if not isinstance(input_df, pd.DataFrame):
            raise TypeError("DataAggregate input must be a pandas DataFrame.")
        if input_df.empty or self.method is None:
            output_df = input_df.copy()
        else:
            working_df = input_df.copy()
            group_column = self.ref
            if self.tag_mapping:
                if self.ref not in working_df:
                    raise ValueError(f"Input table is missing the {self.ref} column.")
                working_df["Tag"] = working_df[self.ref].map(self.tag_mapping)
                group_column = "Tag"

            numeric_columns = working_df.select_dtypes(include=np.number).columns.tolist()
            numeric_columns = [column for column in numeric_columns if column != "Slice"]
            if group_column is None:
                values = getattr(working_df[numeric_columns], self.method)()
                output_df = values.to_frame().T
            else:
                if group_column not in working_df:
                    raise ValueError(f"Input table is missing the {group_column} column.")
                output_df = (
                    working_df.groupby(group_column, dropna=False)[numeric_columns]
                    .agg(self.method)
                    .reset_index()
                )
            output_df = output_df.round(4)

        memory[self.output_df_name] = filter_measurement_columns(
            output_df,
            self.filter_columns,
        )


class DataApply(PipelineAction):
    """Apply a named or callable DataFrame transformation."""

    def __init__(
        self,
        pipeline,
        func: Callable[[pd.DataFrame], pd.DataFrame] | str,
        input_df: str = "tmp/bodycomposition",
        output_df: str = "tmp/bodycomposition",
        filter_columns: str | None = None,
    ):
        super().__init__(pipeline)
        self.func = func
        self.input_df_name = input_df
        self.output_df_name = output_df
        self.filter_columns = filter_columns
        self.io_inputs = [input_df]
        self.io_outputs = [output_df]

    @staticmethod
    def weighted_mean_hu(df: pd.DataFrame) -> pd.DataFrame:
        summary = {}
        for voxel_column in (column for column in df if column.startswith("Vx_")):
            tissue = voxel_column.removeprefix("Vx_")
            hu_column = f"HU_{tissue}"
            if hu_column not in df:
                continue
            value, status = weighted_mean_hu(df[voxel_column], df[hu_column])
            summary[f"HU_weighted_{tissue}"] = value
            summary[f"HU_weighted_status_{tissue}"] = status
        return pd.DataFrame([summary])

    def __call__(self, memory):
        super().__call__(memory)
        input_df = memory[self.input_df_name]
        if not isinstance(input_df, pd.DataFrame):
            raise TypeError("DataApply input must be a pandas DataFrame.")
        if callable(self.func):
            output_df = self.func(input_df.copy())
        elif isinstance(self.func, str) and hasattr(self, self.func):
            output_df = getattr(self, self.func)(input_df.copy())
        else:
            raise ValueError(f"Unknown DataApply function: {self.func!r}.")
        if not isinstance(output_df, pd.DataFrame):
            raise TypeError("DataApply functions must return a pandas DataFrame.")
        memory[self.output_df_name] = filter_measurement_columns(output_df, self.filter_columns)


class DataExport(PipelineAction):
    """Export a pipeline DataFrame to a CSV interoperability file."""

    def __init__(
        self,
        pipeline,
        input_df: str = "tmp/bodycomposition",
        file: str = "exports/{caseid}_raw.csv",
        append: bool = False,
        add_metadata: bool = False,
    ):
        super().__init__(pipeline)
        self.input_df_name = input_df
        self.output_file = file
        self.append = append
        self.add_metadata = add_metadata
        self.timestamp = pipeline.timestamp
        self.io_inputs = [input_df]
        if add_metadata:
            self.io_inputs.append("tmp/metadata")
        self.io_outputs = [file]
        self.io_persisted_outputs = [] if append else [file]
        self.io_reset_outputs = [file]

    def __call__(self, memory):
        super().__call__(memory)
        time_start = time()
        results_df = memory[self.input_df_name]
        if not isinstance(results_df, pd.DataFrame):
            raise TypeError("DataExport input must be a pandas DataFrame.")
        results_df = results_df.copy()

        if self.add_metadata:
            metadata = memory["tmp/metadata"]
            metadata_row = {
                "case_id": memory["id"],
                "case_timestamp": self.timestamp,
                **metadata,
            }
            geometry = memory.get("tmp/tissue_geometry")
            if geometry is not None:
                metadata_row["scan_spacing_z_mm"] = geometry.spacing_xyz[2]
            metadata_df = pd.DataFrame([metadata_row] * len(results_df))
            results_df = pd.concat(
                [metadata_df.reset_index(drop=True), results_df.reset_index(drop=True)],
                axis=1,
            )

        output_path = Path(memory["workspace"]) / self.output_file.format(caseid=memory["id"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        append_existing = self.append and output_path.exists()
        results_df.to_csv(
            output_path,
            mode="a" if append_existing else "w",
            header=not append_existing,
            index=False,
        )
        logging.info(" output: file:%s (%.2fs)", output_path, time() - time_start)
