# Preservation changes and comparability

- Replaced broken framework setup with exact upstream/patch validation and safe repeat use.
- Pinned model download and recorded actual runtime package observations.
- Added archive checksums and raw-result reconstruction; retained old rounded CSV under `legacy/`.
- Aggregate costs now use raw precision: LinearUCB extra tokens display 51.6% instead of 51.7%.
  No rewards, trajectories, seeds, model architecture, losses or probe logic were changed.
- New public runs save a final checkpoint by default; historical main runs saved none.
  This changes storage and checkpoint I/O, not the training algorithm. Use `SAVE_FREQ=-1`
  only when deliberately recreating the historical storage behavior.
- Added future run provenance, CPU/script CI and storage/reproduction documentation.

Archived results were regenerated and matched. Fresh GPU retraining remains unverified.
