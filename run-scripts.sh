# echo "Running p000_preprocess_dataset.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p000_preprocess_dataset.py

# echo "Running p001_preprocess_groundtruth_detection.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p001_preprocess_groundtruth_detection.py

# echo "Running p002_preprocess_groundtruth_tracking.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p002_preprocess_groundtruth_tracking.py

# echo "Running p003_preprocess_groundtruth_visualize.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p003_preprocess_groundtruth_visualize.py

# echo "Running p004_preprocess_train_detectors.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p004_preprocess_train_detectors.py

# echo "Running p010_tune_segment_videos.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p010_tune_segment_videos.py

# echo "Running p011_tune_detect.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p011_tune_detect.py

# echo "Running p012_tune_create_training_data.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p012_tune_create_training_data.py

# echo "Running p013_tune_train_classifier.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p013_tune_train_classifier.py --clear

# echo "Running p014_tune_select_classifier.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p014_tune_select_classifier.py

# echo "Running p015_tune_regulate_tracking.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p015_tune_regulate_tracking.py

# echo "Running p020_exec_classify.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p020_exec_classify.py --clear

# echo "Running p021_exec_classify_correct.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p021_exec_classify_correct.py

# echo "Running p022_exec_classify_visualize.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p022_exec_classify_visualize.py

# echo "Running p023_exec_classify_render.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p023_exec_classify_render.py

# echo "Running p024_exec_classify_tradeoff.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p024_exec_classify_tradeoff.py

echo "Running p030_exec_compress.py"
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p030_exec_compress.py --clear --tilepadding

# echo "Running p031_exec_compress_visualize.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p031_exec_compress_visualize.py

# echo "Running p040_exec_detect.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p040_exec_detect.py --clear

# echo "Running p050_exec_uncompress.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p050_exec_uncompress.py

# echo "Running p060_exec_track.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p060_exec_track.py

# echo "Running p070_accuracy_compute.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p070_accuracy_compute.py --clear

# echo "Running p071_accuracy_visualize.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p071_accuracy_visualize.py

echo "Running p080_throughput_gather.py"
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p080_throughput_gather.py

echo "Running p081_throughput_compute.py"
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p081_throughput_compute.py

echo "Running p082_throughput_visualize.py"
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p082_throughput_visualize.py

# echo "Running p090_tradeoff_compute.py"
# CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p090_tradeoff_compute.py

# echo "Running p091_tradeoff_visualize.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p091_tradeoff_visualize.py

# echo "Running p092_tradeoff_visualize_all.py"
# # CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/p092_tradeoff_visualize_all.py