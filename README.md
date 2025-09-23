# finlab-guard

A lightweight package for managing a local finlab data cache with versioning and time-context features.

![Python versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![Windows](https://img.shields.io/badge/OS-Windows-0078D6?logo=windows&logoColor=white)
![Linux](https://img.shields.io/badge/OS-Linux-FCC624?logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/OS-macOS-000000?logo=apple&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![build](https://github.com/iapcal/finlab-guard/actions/workflows/build.yml/badge.svg)](https://github.com/iapcal/finlab-guard/actions/workflows/build.yml)
[![lint](https://github.com/iapcal/finlab-guard/actions/workflows/lint.yml/badge.svg)](https://github.com/iapcal/finlab-guard/actions/workflows/lint.yml)
[![coverage](https://img.shields.io/codecov/c/github/iapcal/finlab-guard)](https://codecov.io/gh/iapcal/finlab-guard)

## Installation

```bash
pip install finlab-guard
```

## Usage examples

Two short examples showing the most common flows.

### 1) Monkey-patch finlab.data.get (installing FinlabGuard)

This project can monkey-patch `finlab.data.get` so reads go through the guarded cache. Example:

```python
import finlab
from finlab_guard import FinlabGuard

# Create a FinlabGuard instance and install the monkey-patch
guard = FinlabGuard()
guard.install_patch()

# Use finlab.data.get as normal; FinlabGuard will intercept and use cache
result = finlab.data.get('price:收盤價')

# When done, remove the monkey-patch
guard.remove_patch()
```

### 2) Set a time context and get historical data

FinlabGuard supports a time context so you can query data "as-of" a past time.

```python
import finlab
from finlab_guard import FinlabGuard
from datetime import datetime, timedelta
guard = FinlabGuard()
guard.install_patch()

# Set time context to 7 days ago
query_time = datetime.now() - timedelta(days=7)
guard.set_time_context(query_time)

# Now call finlab.data.get normally; the guard will return historical data
result = finlab.data.get('price:收盤價')

# Clear the time context and remove the monkey-patch when done
guard.clear_time_context()
guard.remove_patch()
```

## Performance

finlab-guard delivers significant performance improvements through its DuckDB + Polars architecture:

🚀 **Cache Hit Performance**: 30.6% faster than previous pandas-based implementation

| Version | Cache Hit Time | Improvement |
|---------|---------------|-------------|
| v0.1.0 (pandas.stack) | 17.9 seconds | baseline |
| v0.2.0 (DuckDB+Polars) | 12.4 seconds | **-30.6%** ⚡ |

*Benchmark: `etl:adj_close` cache retrieval (4,533 × 2,645 DataFrame) - average of 10 runs*

### Key Optimizations

- **Eliminated pandas.stack() bottleneck**: Replaced with vectorized Polars operations
- **Cell-level change tracking**: Only stores actual differences, not full datasets
- **DuckDB storage engine**: High-performance indexed storage with time-based reconstruction
- **Intelligent thresholding**: Large row changes stored efficiently as JSON objects

These improvements make finlab-guard ideal for:
- Large datasets with frequent updates
- Historical data analysis and backtesting
- Production environments requiring consistent performance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.