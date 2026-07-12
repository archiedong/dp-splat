# External dependency (not vendored)

The VBGS baseline (VERSES Academic / Nonprofit Research License) is NOT included here. To
run the baseline comparisons and the cross-code regression test, clone it pinned at 2ae3f4be:

    git clone https://github.com/VersesTech/vbgs third_party/vbgs
    git -C third_party/vbgs checkout 2ae3f4be

All dp-splat sources in this repository are MIT-licensed and run without VBGS (only
baseline-comparison scripts and tests/test_vbgs_regression.py need it; that test skips
automatically when the checkout is absent).
