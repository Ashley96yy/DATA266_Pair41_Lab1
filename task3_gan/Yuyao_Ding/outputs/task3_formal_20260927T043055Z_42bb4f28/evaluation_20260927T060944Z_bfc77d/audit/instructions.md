# Independent blind review

Two real people independently score the same fixed cases. Do not show them the run, checkpoint or audit_key.json, and do not discuss scores until both CSVs are complete. The left/source file is the input; translation is the model output. Target style is given in the CSV. This audit is blind to model identity, not to translation direction.

Use integer scores 1-5: style (1=no target style, 3=partial, 5=convincing); content (1=major structure lost, 3=partly preserved, 5=structure preserved); artifact_free (1=severe defects, 3=visible defects, 5=no noticeable defects). Use 2 and 4 for intermediate cases. Higher is better for all three columns. Add the same rater_name to every row of your own CSV. Comments are optional.

The cases were fixed from held-out data before training. Scores are human judgments; blank forms do not count as a completed audit.
