"""Run all tests without pytest: ``python tests/run_tests.py``. (With pytest installed, ``pytest tests`` also works.)"""
import importlib
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    failures = 0
    for path in sorted(HERE.glob('test_*.py')):
        module = importlib.import_module(path.stem)
        for name in sorted(dir(module)):
            if not name.startswith('test_'):
                continue
            t0 = time.time()
            try:
                getattr(module, name)()
                status = 'ok'
            except Exception:                                    # report every failure, then continue
                failures += 1
                status = 'FAILED'
                traceback.print_exc()
            print("%-8s %s.%s (%.1f s)" % (status, path.stem, name, time.time() - t0), flush=True)
    print("%d failure(s)" % failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
