rsync -avm --dry-run \
  --include='*/' \
  --include='*/test/' \
  --include='*/test/video/***' \
  --include='*/test/yolov3-*/***' \
  --include='*/test/gt*' \
  --include='*/test/gt*/***' \
  --exclude='*' \
  /work/cwkt/data/otif-dataset/dataset/ \
  /work/cwkt/data/polyis-data/sources2/