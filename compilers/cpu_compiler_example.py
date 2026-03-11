from mlir import ir
from mlir.passmanager import PassManager

from lighthouse.pipeline.helper import PassBundles, add_bundle

from mlir.dialects import transform

from lighthouse.pipeline.helper import apply_registered_pass, canonicalize, match



    def payload_module(self) -> ir.Module:
        mod = ir.Module.create()

        with ir.InsertionPoint(mod.body):
            float32_t = ir.F32Type.get()
            shape = (self.M, self.N)
            tensor_t = ir.RankedTensorType.get(shape, float32_t)
            memref_t = ir.MemRefType.get(shape, float32_t)
            fargs = [memref_t, memref_t, memref_t]

            @func.func(*fargs, name=self.payload_function_name)
            def payload(A, B, C):
                a_tensor = bufferization.to_tensor(tensor_t, A, restrict=True)
                b_tensor = bufferization.to_tensor(tensor_t, B, restrict=True)
                c_tensor = bufferization.to_tensor(
                    tensor_t, C, restrict=True, writable=True
                )
                add = linalg.add(a_tensor, b_tensor, outs=[c_tensor])
                bufferization.materialize_in_destination(
                    None, add, C, restrict=True, writable=True
                )

        payload.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return mod

    def schedule_module(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> ir.Module:
        schedule_module = ir.Module.create()
        schedule_module.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )
        with ir.InsertionPoint(schedule_module.body):
            named_sequence = transform.named_sequence(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )
            with ir.InsertionPoint(named_sequence.body):
                anytype = transform.AnyOpType.get()
                func = match(named_sequence.bodyTarget, ops={"func.func"})
                mod = transform.get_parent_op(
                    anytype,
                    func,
                    op_name="builtin.module",
                    deduplicate=True,
                )
                mod = apply_registered_pass(mod, "one-shot-bufferize")
                mod = apply_registered_pass(mod, "convert-linalg-to-loops")
                transform.apply_cse(mod)
                canonicalize(mod)

                if stop_at_stage == "bufferized":
                    transform.YieldOp()
                    return schedule_module

                mod = apply_registered_pass(mod, "convert-scf-to-cf")
                mod = apply_registered_pass(mod, "finalize-memref-to-llvm")
                mod = apply_registered_pass(mod, "convert-cf-to-llvm")
                mod = apply_registered_pass(mod, "convert-arith-to-llvm")
                mod = apply_registered_pass(mod, "convert-func-to-llvm")
                mod = apply_registered_pass(mod, "reconcile-unrealized-casts")
                transform.YieldOp()

        return schedule_module


if __name__ == "__main__":
    with ir.Context(), ir.Location.unknown():
        wload = ElementwiseSum(400, 400)

        print(" Dump kernel ".center(60, "-"))
        wload.lower_payload(dump_payload="bufferized", dump_schedule=True)

        print(" Execute 1 ".center(60, "-"))
        execute(wload, verbose=2)

        print(" Execute 2 ".center(60, "-"))
        execute(wload, verbose=1)

        print(" Benchmark ".center(60, "-"))
        times = benchmark(wload)
        times *= 1e6  # convert to microseconds
        # compute statistics
        mean = np.mean(times)
        min = np.min(times)
        max = np.max(times)
        std = np.std(times)
        print(f"Timings (us): mean={mean:.2f}+/-{std:.2f} min={min:.2f} max={max:.2f}")
        flop_count = wload.get_complexity()[0]
        gflops = flop_count / (mean * 1e-6) / 1e9
        print(f"Throughput: {gflops:.2f} GFLOPS")
