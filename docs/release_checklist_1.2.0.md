# Release Checklist 1.2.0

## Scope status

- WM-1 completed: CRS-aware terrain slope spacing and tests.
- WM-2 completed: AWEInsh SWIR2 formula parity across local and GEE paths with tests.
- WM-3 completed: Hydroperiod denominator corrected for valid observations; empty years excluded.
- WM-4 completed: Offline climate-adaptive decision-logic tests added; weekly live-GEE workflow added.
- WM-5 completed: dead property-level climate filter removed.
- WM-6 completed: assert-based invariant replaced with RuntimeError and NaN-safe validation.
- WM-7 completed: optional min_support guard for New/Lost with tests.
- WM-8 completed: documentation corrections for adaptivity wording and limitations.
- WM-9 completed (hardening item addressed): warning on high-NaN use of nan_policy='total'.

## Validation evidence

- Full test suite: 226 passed, 10 skipped, 0 failed.
- Live GEE tests remain opt-in and are skipped by default in local runs.

## Release preflight

- Ensure live GEE weekly workflow secrets are configured:
  - WETLANDMAPPER_GEE_SERVICE_ACCOUNT
  - WETLANDMAPPER_GEE_PROJECT
- Review warnings from terrain tests (expected no-CRS heuristic warnings).
- Confirm CHANGELOG section [Unreleased 1.2.0] reflects any final deltas.
- Keep notebook edits out of release commits unless intentionally included.

## Tagging steps

1. Verify clean staged content (exclude notebooks unless requested).
2. Run final test suite: `python -m pytest -q`.
3. Commit any remaining release-note/doc changes.
4. Create tag: `git tag v1.2.0`.
5. Push tag: `git push origin v1.2.0`.
6. Publish GitHub release notes from CHANGELOG highlights.
