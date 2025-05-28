from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    # Create a benchmark context without parameters
    context = BenchmarkRegistry.create_benchmark_context(
        'nvbench_kernel_launch',
        platform=Platform.CUDA
    )

    # Launch the benchmark
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )