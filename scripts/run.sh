#!/bin/bash
# Two-pass pipeline: valid pass → evaluation + Pareto extraction → test pass → final evaluation

# echo -e "\n\n\n=============================================================="
# echo          "Phase 0: Test pass (learning on test data)"
# echo -e       "==============================================================\n"
# # echo -e "\n\n\n########################################"
# # echo ">>> scripts/p015_tune_optimize_classifier.py"
# echo -e "########################################\n"
# python scripts/p015_tune_optimize_classifier.py
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p016_tune_track_rate.py"
# echo -e "########################################\n"
# python scripts/p016_tune_track_rate.py

# echo -e "\n\n\n=============================================================="
# echo          "Phase 1: Valid pass (full parameter grid, valid videoset only)"
# echo -e       "==============================================================\n"

# echo -e "\n\n\n########################################"
# echo ">>> scripts/p020_exec_classify.py"
# echo -e "########################################\n"
# python scripts/p020_exec_classify.py --valid
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p022_exec_prune_polyominoes.py"
# echo -e "########################################\n"
# python scripts/p022_exec_prune_polyominoes.py --valid
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p030_exec_compress.py"
# echo -e "########################################\n"
# python scripts/p030_exec_compress.py --valid
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p040_exec_detect.py"
# echo -e "########################################\n"
# python scripts/p040_exec_detect.py --valid
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p050_exec_uncompress.py"
# echo -e "########################################\n"
# python scripts/p050_exec_uncompress.py --valid
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p060_exec_track.py"
# echo -e "########################################\n"
# python scripts/p060_exec_track.py --valid

# echo -e "\n\n\n########################################"
# echo ">>> preprocess/p002_preprocess_naive_tracking.py --valid"
# echo -e "########################################\n"
# python preprocess/p002_preprocess_naive_tracking.py --valid

# echo -e "\n\n\n==================================================================="
# echo          "Phase 2+3: Evaluation on valid data and Pareto parameter extraction"
# echo -e       "===================================================================\n"

# echo -e "\n\n\n########################################"
# echo ">>> evaluation/p100_evaluation_val.py"
# echo -e "########################################\n"
# python evaluation/p100_evaluation_val.py



# echo -e "\n\n\n====================================================================="
# echo          "Phase 4: Test pass (auto-filtered to Pareto-optimal parameter combos)"
# echo -e       "=====================================================================\n"

# echo -e "\n\n\n########################################"
# echo ">>> scripts/p020_exec_classify.py"
# echo -e "########################################\n"
# python scripts/p020_exec_classify.py --test
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p022_exec_prune_polyominoes.py"
# echo -e "########################################\n"
# python scripts/p022_exec_prune_polyominoes.py --test
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p030_exec_compress.py"
# echo -e "########################################\n"
# python scripts/p030_exec_compress.py --test
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p040_exec_detect.py"
# echo -e "########################################\n"
# python scripts/p040_exec_detect.py --test
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p050_exec_uncompress.py"
# echo -e "########################################\n"
# python scripts/p050_exec_uncompress.py --test
# echo -e "\n\n\n########################################"
# echo ">>> scripts/p060_exec_track.py"
# echo -e "########################################\n"
# python scripts/p060_exec_track.py --test

# echo -e "\n\n\n########################################"
# echo ">>> preprocess/p002_preprocess_naive_tracking.py --test"
# echo -e "########################################\n"
# python preprocess/p002_preprocess_naive_tracking.py --test
# echo -e "\n\n\n########################################"
# echo ">>> preprocess/p003_preprocess_groundtruth_tracking.py --test"
# echo -e "########################################\n"
# python preprocess/p003_preprocess_groundtruth_tracking.py --test

echo -e "\n\n\n======================================================="
echo          "Phase 5: Final evaluation (test data for Pareto combos)"
echo -e       "=======================================================\n"

echo -e "\n\n\n########################################"
echo ">>> evaluation/p101_evaluation_test.py"
echo -e "########################################\n"
python evaluation/p101_evaluation_test.py
echo -e "\n\n\n########################################"
echo ">>> evaluation/p203_compare_stats.py --test"
echo -e "########################################\n"
python evaluation/p203_compare_stats.py --test
