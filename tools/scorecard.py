#!/usr/bin/env python3
"""One table for the engine homework: each spec test, pass or fail, and the Hull
equation it is waiting on. Run it after every function you write.

    python3 tools/scorecard.py

The engine is deterministic, so this is pytest with a different face. The same
four-column habit (case, result, what failed, where the truth lives) is what the
agent evals use once there is a model in the loop.
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

ORDER = ['d1_d2', 'bs_price', 'greeks', 'implied_vol', 'binomial_price']


class Collector:
    def __init__(self):
        self.rows = []

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if report.when != 'call':
            return
        doc = (item.function.__doc__ or '').strip().splitlines()
        hint = doc[0] if doc else ''
        waiting_on = ''
        if report.failed and call.excinfo is not None:
            exc = call.excinfo.value
            if isinstance(exc, NotImplementedError):
                frame = call.excinfo.traceback[-1]
                waiting_on = f'{frame.name}: {exc}'
            else:
                waiting_on = f'{type(exc).__name__}: {str(exc).splitlines()[0][:70]}' if str(exc) else type(exc).__name__
        self.rows.append({'test': item.name, 'passed': report.passed, 'hint': hint, 'waiting_on': waiting_on})


def main():
    collector = Collector()
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main(['-q', '-q', '-p', 'no:cacheprovider', '--no-header', '--tb=no', '--no-summary',
                     str(ROOT / 'tests' / 'test_engine.py')], plugins=[collector])
    rows = collector.rows
    width = max(len(r['test']) for r in rows) + 2
    print()
    print(f'{"test":{width}}{"result":8}waiting on')
    print('-' * (width + 8 + 60))
    for r in rows:
        print(f'{r["test"]:{width}}{"pass" if r["passed"] else "FAIL":8}{r["waiting_on"]}')
        if r['hint'] and not r['passed']:
            print(f'{"":{width}}        {r["hint"][:110]}')
    print('-' * (width + 8 + 60))
    passed = sum(r['passed'] for r in rows)
    waiting = {}
    for r in rows:
        m = re.match(r'(\w+):', r['waiting_on'])
        if m and not r['passed']:
            waiting.setdefault(m[1], 0)
            waiting[m[1]] += 1
    order = [f for f in ORDER if f in waiting] + [f for f in waiting if f not in ORDER]
    print(f'engine spec: {passed}/{len(rows)} passing' + ('' if not waiting else
          '   next: ' + ', '.join(f'{f} (unblocks {waiting[f]})' for f in order)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
