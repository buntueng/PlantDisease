ChiliLite-V2 — Final Edited Code
=================================

PROJECT PATH
------------
/home/bt/Documents/Po/chili_fruit/dataset/Chili Growth Stage Original Dataset (1)/Chili Growth Stage Original Dataset

FILES
-----
extract_chili_dataset.py
check_dataset_manifest.py
run_baseline_models.py
run_proposed_chililite_v2.py
visualize_chililite_v2_results.ipynb
requirements.txt

IMPORTANT
---------
The original folders remain:
Dry Chili 410, Flower 397, Green Chili 328, Red Chili 200, Rotten Chili 379.

After each 10-fold split, only the inner-training data are balanced to:
410 samples x 5 classes = 2,050 training samples per fold.
Validation and test data are never augmented or oversampled.

INSTALL
-------
cd "/home/bt/Documents/Po/chili_fruit/dataset/Chili Growth Stage Original Dataset (1)/Chili Growth Stage Original Dataset"
/bin/python3 -m pip install -r requirements.txt

STEP 1 — DELETE THE OLD INCORRECT MANIFEST
-------------------------------------------
rm -f data/chili_growth_stage/dataset_manifest.csv
rm -f data/chili_growth_stage/dataset_info.json
rm -f data/chili_growth_stage/invalid_images.csv

STEP 2 — REBUILD THE MANIFEST
------------------------------
/bin/python3 extract_chili_dataset.py --workdir "$PWD" --force

Because path defaults are now relative to the script, this also works:
/bin/python3 "/home/bt/Documents/Po/chili_fruit/dataset/Chili Growth Stage Original Dataset (1)/Chili Growth Stage Original Dataset/extract_chili_dataset.py"

EXPECTED ORIGINAL COUNTS
------------------------
Dry Chili: 410
Flower: 397
Green Chili: 328
Red Chili: 200
Rotten Chili: 379
Total: 1714

STEP 3 — CHECK BEFORE TRAINING
-------------------------------
/bin/python3 check_dataset_manifest.py

The manifest must contain IDs 0, 1, 2, 3, 4. ID 3 is Red Chili.

STEP 4 — FAIR JOINT TRAINING
-----------------------------
/bin/python3 run_proposed_chililite_v2.py \
  --with-baselines \
  --image-size 224 \
  --target-per-class 410 \
  --epochs 45 \
  --batch-size 32 \
  --num-workers 0 \
  --measure-latency

OUTPUT
------
results/fair_comparison/all_fold_metrics.csv
results/fair_comparison/summary_metrics.csv
results/fair_comparison/all_predictions.csv
results/fair_comparison/model_complexity.csv
results/fair_comparison/split_assignments.csv
results/fair_comparison/experiment_config.json

EXPECTED BALANCED CONSOLE OUTPUT
--------------------------------
Dry Chili: 410
Flower: 410
Green Chili: 410
Red Chili: 410
Rotten Chili: 410
Total balanced training samples: 2050
