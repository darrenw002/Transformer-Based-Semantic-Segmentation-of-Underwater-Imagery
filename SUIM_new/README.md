# SUIM cleanup export

train_val/images contains byte-identical copies of retained source images.
train_val/masks contains aligned RGB PNG masks; source label colors are preserved.
Optional masks_class_id PNGs contain class IDs, not RGB annotations.
Color policy: preserve. Consult reports/class_definition.json.
Ignore ID 255, if present, must be excluded from loss and metrics.
Threshold decoding is a color mapping, not a verified semantic repair.
TEST was inventoried only and is not copied here; use the unchanged original TEST split.
reports/pairs.csv logs all source pairs and exclusions.
reports/duplicate_groups.csv is also a review template: fill keep_stem, then pass it
with --duplicate-decisions when exporting to a new output directory.
Blank keep_stem retains all group members. Keep each group in one data partition.
Read reports/summary.json for unresolved colors and retained duplicates.
