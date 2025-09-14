#!/usr/bin/env python3

from roop import core

if __name__ == '__main__':
    core.run()

"""

python run.py --execution-provider cpu \
-s /Users/kusch/PycharmProjects/roop_colab/1.jpg \
-t '/Users/kusch/PycharmProjects/roop_colab/4.jpg' \
-o '/Users/kusch/PycharmProjects/roop_colab/hahahahaha.jpg' \
--frame-processor face_swapper face_enhancer \
--output-video-encoder libx264 \
--output-video-quality 35 \
--keep-fps \
--many-faces \
--temp-frame-format jpg \
--temp-frame-quality 0

"""