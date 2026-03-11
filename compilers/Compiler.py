from mlir import ir
from mlir.passmanager import PassManager

from lighthouse.pipeline.helper import PassBundles, add_bundle

import ctypes
from contextlib import contextmanager
from functools import cached_property
from typing import Optional

import numpy as np
from mlir import ir
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from mlir.dialects import func, linalg, bufferization
from mlir.dialects import transform
from mlir.execution_engine import ExecutionEngine

from lighthouse.pipeline.helper import apply_registered_pass, canonicalize, match
from lighthouse.workload import Workload, execute, benchmark


class Compiler:
    """
    A simple compiler that applies a predefined set of transformations to lower the payload module to LLVM IR dialect.

    This compiler can be used as a reference for how to implement the compilation pipeline for a workload. It can also be used as a baseline for performance comparison with more advanced compilation strategies.
    """
    def __init__(self):
        self.payload_module = None

    def import_payload(self, payload_module: ir.Module) -> None:
        self.payload_module = payload_module

    def pass_bundle(self, bundle: list[str]) -> None:
        if self.payload_module is None:
            raise ValueError("Payload module is not imported yet.")
        pm = PassManager("builtin.module", self.payload_module.context)
        add_bundle(pm, bundle)
        pm.run(self.payload_module.operation)

    def bufferize(self) -> None:
        bufferization_pipeline = PassBundles.BufferizationBundle + PassBundles.CleanupBundle
        self.pass_bundle(bufferization_pipeline)

    def lower_to_llvm(self) -> None:
        lowering_pipeline = PassBundles.LLVMLoweringBundle + PassBundles.CleanupBundle
        self.pass_bundle(lowering_pipeline)
