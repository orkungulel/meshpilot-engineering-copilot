# MeshPilot demo notes: what the model predicts, its limits, and how to review it

**Status:** project notes for the MeshPilot demo, written by the project
author.
- MeshPilot is a research prototype. Its predictions are not validated for
  engineering decisions.
- These notes are not an engineering standard, not a company procedure and
  not an automatic compliance check.
- A prediction is never a compliance decision, and nothing here certifies a
  part.

**Terms:**
- A *B-rep face* is a face of a CAD model's boundary representation.
- A *triangle* is one element of the triangle mesh made from the model. One
  B-rep face usually becomes many triangles.
- MeshPilot predicts one class per triangle. In the viewer and in the
  geometry tool's output, "faces" means mesh triangles.

**Numbers:** every figure in sections 2 to 4 is filled in when this document
is built, from the records written by `meshpilot.verify` and
`meshpilot.predict`.
- The build checks those records against each other and against the files
  they describe.
- It stops if any figure is missing.
- The sources list identifies each record by its SHA-256.

## 1. What the model predicts

- **How the dataset labels faces:** in the Fusion 360 Gallery segmentation
  dataset, each B-rep face is labelled with the modelling operation that
  created it, taken from the design's parametric feature timeline.
- **Which operations:** only extrude, chamfer, fillet and revolve
  operations are labelled. Other CAD features were suppressed in the
  dataset's designs [S1 "Overview", "Other Notes"].
- **Triangles:** each triangle of a dataset mesh carries the label of the
  B-rep face it belongs to [S1 "Mesh"]. The model learned to predict that
  label for every triangle.

| Label | Class | Meaning (paraphrased from S1 "Labels") |
|---|---|---|
| 0 | ExtrudeSide | a side B-rep face of an extrude that adds material |
| 1 | ExtrudeEnd | a start or end B-rep face of an extrude that adds material |
| 2 | CutSide | a side B-rep face of an extrude that removes material |
| 3 | CutEnd | a start or end B-rep face of an extrude that removes material |
| 4 | Fillet | a B-rep face made by a fillet |
| 5 | Chamfer | a B-rep face made by a chamfer |
| 6 | RevolveSide | a side B-rep face of a revolve |
| 7 | RevolveEnd | a start or end B-rep face of a revolve |

The label numbers follow the dataset's `segment_names.json` [S2].

The classes say how a B-rep face was modelled. They do not say what the face
is for, or how it would be manufactured:
- CutSide does not mean "hole";
- Fillet says nothing about a machining process;
- no class is a machining feature.

## 2. The model, its checkpoint and its data

- **Model:** DiffusionNet [S3], trained in the author's MSc thesis [S4]:
  - {{input_features}} input: the triangle mesh's vertex positions, centred
    and scaled;
  - {{width}} channels, {{blocks}} blocks and {{k_eig}} eigenvectors;
  - {{n_class}} classes.
- **Checkpoint:** the last of its training run, epoch {{epochs_completed}}
  of {{epochs_planned}}. {{selection}} SHA-256 {{checkpoint_sha}}. [S4]
- **Data:** a subset of the Fusion 360 Gallery segmentation meshes, divided
  by the thesis's recorded split into {{train_meshes}} training and
  {{test_meshes}} test meshes. The split file's SHA-256 is {{split_sha}}.
  This is not the dataset's official split [S1 "Train/Test Split"]. [S5]
- **Augmentation:** training used no rotation or other augmentation. [S4]

## 3. Test-split results (benchmark context: all {{test_meshes}} test meshes)

**Provenance:**
- Checkpoint {{checkpoint_sha}}, split file {{split_sha}}.
- Evaluation by `meshpilot.verify`, record {{verify_record_sha}} [S6]. It
  reran the checkpoint on the training run's cached copy of the test meshes
  and compared the result with the scores the run recorded for the same
  epoch. They agree within {{score_tolerance}}.

**How the scores are counted:**
- **One confusion matrix** over every labelled triangle of all
  {{test_meshes}} test meshes.
- **Per-class IoU** is TP / (TP + FP + FN).
- **Absent classes:** a class with no labelled and no predicted triangle has
  no IoU and is left out of the mean. A class that is labelled but never
  predicted, or predicted but never labelled, has IoU 0 and is included.
- **Mean IoU** is the unweighted mean of the per-class IoUs.
- **Triangle accuracy** is the share of labelled triangles predicted
  correctly.

The results:
- **Mean IoU:** {{test_miou}} (recorded during training: {{recorded_miou}}).
- **Triangle accuracy:** {{test_accuracy}} (recorded: {{recorded_accuracy}}).

{{test_split_iou_table}}

These figures describe the test split as a whole. They are not a confidence
for any triangle of any particular part, and they are not the results of the
reference mesh in section 4.

## 4. Fixed reference mesh {{mesh_id}} (one mesh, an illustration)

These numbers belong to one fixed reference mesh, {{mesh_id}}, and to the
prediction recorded for it. They are not about whatever mesh is open in the
viewer:
- numbers about the current mesh come only from the geometry tool;
- this section applies only when the current mesh's ID is {{mesh_id}}.

**Provenance:**
- Checkpoint {{checkpoint_sha}}.
- The mesh has {{mesh_triangles}} triangles, SHA-256 {{mesh_sha}}. It is in
  the test list of split file {{split_sha}}, so it was not used in training.
- Prediction by `meshpilot.predict`, record {{prediction_record_sha}} [S7].

**How the scores are counted:** as in section 3, but over this mesh's
labelled triangles only. So a class absent from both this mesh's labels and
its prediction is left out: {{mesh_classes_absent}}.

{{mesh_class_table}}

- **Triangle accuracy:** {{mesh_accuracy}}.
- **Mean IoU:** {{mesh_miou}}, over the {{mesh_classes_present}} classes
  counted.

Scores on one mesh illustrate one prediction. They are not the model's
benchmark performance.

## 5. Limitations

- **A research prototype.** It is not validated for engineering decisions.
- **Predictions are not compliance decisions.** No prediction, patch or
  score shows that a part meets a requirement.
- **Accuracy is low overall and uneven by class.** The lowest test-split IoU
  is for {{lowest_iou_classes}} (section 3).
- **Whole regions can be wrong.** See the reference mesh's table in
  section 4.
- **No calibrated confidence.** The model gives one class per triangle, with
  no reliable measure of how sure it is.
- **Triangles, not B-rep faces.** The model sees only the triangle mesh's
  vertex positions. It does not see B-rep faces, the CAD history,
  dimensions, tolerances, materials or intent. The triangles of one B-rep
  face can receive different classes.
- **Narrow training conditions.** It was trained without augmentation, on
  meshes from one conversion of one dataset. Other poses, tessellations,
  sizes and sources are untested.
- **Unseen features.** Triangles from other CAD features (for example sweeps,
  lofts or shells) were not in the training data. The model still gives
  them one of the eight classes.
- **Patches are not feature instances.**
  - A patch is a group of edge-connected triangles with the same predicted
    class.
  - It is neither a B-rep face nor a feature. One feature can be split into
    several patches, and touching features can merge into one.
  - A patch count is not a count of holes, fillets or any other feature.

## 6. Example human-review steps (an example only, not a procedure)

These steps illustrate how a person might check a prediction in the viewer.
They are not a required, validated or approved method.

1. If the mesh has dataset labels, switch between "Prediction" and "Ground
   truth" and note where they differ.
2. Isolate the class in question and look at each highlighted patch. Note
   any patch that covers geometry the class does not describe, or that
   splits one surface into several classes.
3. Treat the low-IoU classes in section 3 with extra care. The absence of a
   class in the prediction proves nothing.
4. Confirm any engineering conclusion against the CAD model or drawing,
   never against the prediction alone.
5. Write down what you checked. Do not record a prediction or a patch count
   as a measurement.

## 7. Rules for answers that use these notes

1. **Two kinds of evidence, kept apart.** Document evidence is cited text
   from these notes. Model evidence is the geometry tool's output for the
   current mesh. Never present one as the other.
2. **Numbers about the current mesh** come only from the geometry tool.
   Section 4 describes the fixed reference mesh {{mesh_id}} only.
3. **What to cite these notes for:** class meanings, limitations,
   provenance and the review example. Never cite them as proof that a part
   is right or wrong.
4. **Never describe:**
   - test-split IoU as a confidence;
   - a patch count as a feature count;
   - a triangle as a B-rep face;
   - a prediction as a compliance decision.
5. **No manufacturing claims.** Never turn a class into a machining feature,
   a process or a manufacturing plan.
6. **No answer:** if neither these notes nor the tool can answer, say so.

## Sources

- **[S1]** Autodesk AI Lab, "Segmentation Dataset", Fusion 360 Gallery
  Dataset documentation: [segmentation.md @ aca90ae](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/aca90ae537c58925d4304f12b210422a3e23ceb6/docs/segmentation.md).
  Pinned to commit aca90ae537c58925d4304f12b210422a3e23ceb6 and read on 11
  October 2026. The sections used are named in the text.
- **[S2]** `segment_names.json`, which maps segment indices to class names,
  as described in [segmentation.md, B-Rep section](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/aca90ae537c58925d4304f12b210422a3e23ceb6/docs/segmentation.md#b-rep).
- **[S3]** N. Sharp, S. Attaiki, K. Crane and M. Ovsjanikov, "DiffusionNet:
  Discretization Agnostic Learning on Surfaces":
  [arXiv:2012.00888v3](https://arxiv.org/abs/2012.00888v3).
- **[S4]** The thesis training run's recorded configuration and training
  code (not public). Checkpoint SHA-256 {{checkpoint_sha}}.
- **[S5]** The thesis's recorded train/test split of the subset (not
  public). Split file SHA-256 {{split_sha}}.
- **[S6]** The `meshpilot.verify` record, SHA-256 {{verify_record_sha}}.
- **[S7]** The `meshpilot.predict` record, SHA-256 {{prediction_record_sha}}.

The Fusion 360 Gallery data is licensed for non-commercial research use.
These notes quote no dataset content beyond the class names.
