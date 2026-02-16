STRAIGHTFORWARD:

Command to run the "original" model to train a video based on a label

python -m symbiote.cli.main train --video ./videos/5293.mov --label "f" --threshold 10 --image-dir ../images/image-testing --frame-skip 4 --output-dir ../models/classifier

Command to run the HMM model

python -m symbiote.cli.main train \
    --video ../videos/GX011230.mp4 \
    --label "n" \
    --threshold 10 \
    --image-dir ../images/image-testing \
    --train-hmm \
    --annotation ../videos/GX011230_annotations.csv \
    --aruco-config ../config/aruco_bins.json