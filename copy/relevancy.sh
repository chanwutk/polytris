rsync -av \
  --exclude="jnc*" \
  --exclude="*/execution/*/033_compressed_frames" \
  --exclude="*/evaluation" \
  --exclude="*/indexing" \
  --exclude="*/execution/*/040_*" \
  --exclude="*/execution/*/050_*" \
  --exclude="*/execution/*/060_*" \
  --exclude="*/execution/*/020_relevancy/*/statistics" \
  --exclude="*/execution/*/020_relevancy/*/visualization*" \
  --exclude="*/execution/*/000_groundtruth/*.mp4" \
  --exclude="*/execution/va*" \
  --exclude="*/execution/tr*"  \
  polyis-cache /work/chanwutk/data/polyis-cache