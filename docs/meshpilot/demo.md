# MeshPilot demo (about 3 minutes)

## Before the demo

1. Run `predict` and `verify` ([docs/meshpilot/usage.md](usage.md)).
2. For the chat, put `ANTHROPIC_API_KEY` in `.env`.
3. Start the viewer:
   `.venv\Scripts\python -m uvicorn meshpilot.api:app --port 8001`.
4. Open <http://127.0.0.1:8001>.

## Walkthrough

1. **The part.** It is mesh `23910_316134bd_3`, from the recorded test
   split, so the model never trained on it.
   - Drag to rotate it.
   - Point out the "Real model prediction" badge.
2. **Predictions against the dataset labels.**
   - Switch between "Predicted classes" and "Dataset labels".
   - The model labels most of the part ExtrudeSide, and it misses the
     RevolveSide band entirely.
   - Its weak spots are shown, not hidden.
3. **One class.**
   - Click CutEnd in the legend.
   - The other faces turn translucent, revealing an inside floor that the
     outer surface hides.
4. **Where the numbers come from.** Show the provenance panel:
   - the checkpoint, its epoch and hash;
   - the training-time copy of `diffusion_net`.

   Then show the test-split context. `meshpilot.verify` reproduced the
   run's recorded test mIoU from the training cache, and the panel says that
   this IoU is not a confidence.
5. **The chat.** Ask "Show the faces predicted as Fillet".
   - The highlight comes from the tool's structured result.
   - The answer separates "From the model's predictions" from "From the
     documents".
   - Patches are not feature counts.
6. **Limitations.** Read the panel:
   - predictions, not ground truth;
   - no rotation augmentation in training;
   - a local demo, not a production system.

## Do not claim

- That test-split IoU is a confidence for this part.
- That patches are counts of holes or other features.
- That Fusion 360 classes are machining features.
- That the segmentation produces a machining plan.
- That this demo is a production deployment.
