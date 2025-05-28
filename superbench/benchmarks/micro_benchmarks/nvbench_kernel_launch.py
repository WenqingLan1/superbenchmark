import os
import re
from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke

def parse_time_to_us(raw: str) -> float:
    """Helper: parse '123.45 us', '678.9 ns', '0.12 ms' → float µs."""
    raw = raw.strip()
    if raw.endswith('%'):
        return float(raw[:-1])
    # split “value unit” or “valueunit”
    m = re.match(r'([\d.]+)\s*([mun]?s)?', raw)
    if not m:
        return float(raw)
    val, unit = float(m.group(1)), (m.group(2) or 'us')
    if unit == 'ns':  return val / 1e3
    if unit == 'ms':  return val * 1e3
    return val

class NvbenchKernelLaunch(MicroBenchmarkWithInvoke):
    """Nvbench benchmark wrapper for SuperBench."""
    def __init__(self, name, parameters=None):
        """Initialize the benchmark."""
        super().__init__(name, parameters)
        self._bin_name = "nvbench_kernel_launch"

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.
        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)
        self._commands = [f"{self.__bin_path}"]

        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_output, self._args.log_raw_data)
        try:
            gpu_section = r"### \[(\d+)\] NVIDIA"
            row_pat = (
                r"\| (\d+)x \| ([\d.]+ ?[mun]?s) \| ([\d.]+%) \| "
                r"([\d.]+ ?[mun]?s) \| ([\d.]+%) \| (\d+)x \| ([\d.]+ ?[mun]?s) \|"
            )
            current = None
            for line in raw_output.splitlines():
                line = line.strip()
                g = re.match(gpu_section, line)
                if g:
                    current = f"gpu_{g.group(1)}"
                    continue
                r = re.match(row_pat, line)
                if r and current:
                    self._result.add_result(f"{current}_samples", int(r.group(1)))
                    self._result.add_result(f"{current}_cpu_time", parse_time_to_us(r.group(2)))
                    self._result.add_result(f"{current}_cpu_noise", float(r.group(3)[:-1]))
                    self._result.add_result(f"{current}_gpu_time", parse_time_to_us(r.group(4)))
                    self._result.add_result(f"{current}_gpu_noise", float(r.group(5)[:-1]))
                    self._result.add_result(f"{current}_batch_samples", int(r.group(6)))
                    self._result.add_result(f"{current}_batch_gpu_time", parse_time_to_us(r.group(7)))
            if not self._result.result:
                raise RuntimeError("no valid rows parsed")
        except Exception as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                f"invalid result format - round:{self._curr_run_index}, bench:{self._name}, msg:{e}\n{raw_output}"
            )
            return False
        return True

# Register the benchmark
BenchmarkRegistry.register_benchmark("nvbench_kernel_launch", NvbenchKernelLaunch, platform=Platform.CUDA)