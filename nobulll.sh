STRAIGHTFORWARD:

Command to run the "original" model to train a video based on a label

python -m symbiote.cli.main train --video ./videos/5293.mov --label "f" --threshold 10 --image-dir ../images/image-testing --frame-skip 4 --output-dir ../models/classifier