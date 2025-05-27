from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke

class NvbenchKernelLaunch(MicroBenchmarkWithInvoke):
    """Nvbench benchmark wrapper for SuperBench."""
    def __init__(self, name, parameters=None):
        """Initialize the benchmark."""
        super().__init__(name, parameters)
        self._bin_name = "nvbench_kernel_launch"

    def _preprocess(self):
        """Preprocess and validate the benchmark parameters."""
        # Add any necessary preprocessing logic here
        return True

    def _benchmark(self):
        """Run the benchmark."""
        command = f"{self._bin_name} {self._parameters}"
        return self._run_command(command)

    def _process_raw_result(self, cmd_idx, raw_output):
        """Process the raw output from the benchmark.

        Args:
            cmd_idx (int): The index of the command corresponding with the raw_output.
            raw_output (str): Raw output string of the micro-benchmark.

        Returns:
            bool: True if the raw output string is valid and results can be extracted.
        """
        # Store raw data
        self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_output, self._args.log_raw_data)

        try:
            # Regular expressions to extract metrics
            gpu_section_pattern = r"### \[(\d+)\] NVIDIA (\S+)"
            table_row_pattern = r"\| (\d+)x \| ([\d.]+ \w+) \| ([\d.]+%) \| ([\d.]+ \w+) \| ([\d.]+%) \| (\d+)x \| ([\d.]+ \w+) \|"

            # Parse the raw output
            current_gpu = None
            output_lines = raw_output.splitlines()
            for line in output_lines:
                line = line.strip()  # Strip leading and trailing spaces
                gpu_match = re.match(gpu_section_pattern, line)
                if gpu_match:
                    current_gpu = f"gpu_{gpu_match.group(1)}"
                    continue

                row_match = re.match(table_row_pattern, line)
                if row_match and current_gpu:
                    # Extract metrics from the matched row
                    self._result.add_result(f"{current_gpu}_samples", int(row_match.group(1)))
                    self._result.add_result(f"{current_gpu}_cpu_time", row_match.group(2))
                    self._result.add_result(f"{current_gpu}_cpu_noise", row_match.group(3))
                    self._result.add_result(f"{current_gpu}_gpu_time", row_match.group(4))
                    self._result.add_result(f"{current_gpu}_gpu_noise", row_match.group(5))
                    self._result.add_result(f"{current_gpu}_batch_samples", int(row_match.group(6)))
                    self._result.add_result(f"{current_gpu}_batch_gpu_time", row_match.group(7))

            # Check if any results were added
            if not self._result.result:
                raise BaseException("No valid results found.")

        except BaseException as e:
            # Handle parsing errors
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True

# Register the benchmark
BenchmarkRegistry.register_benchmark("nvnvbench_kernel_launch", NvbenchKernelLaunch, Platform.CUDA)