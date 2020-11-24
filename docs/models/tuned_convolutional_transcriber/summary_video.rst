:orphan:

Tuned convolutional transcriber: training summary video
=======================================================

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/442088674?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    <tr><td><center>Three quality metrics for a deep learning model trained to transcribe text data</center></td></tr>
    </table>
    </center>

On the left, Probability distribution of digits from the test dataset, as transcribed by the model, partitioned by correct answer (vertical axis) and transcribed value (horizontal axis). That is, the bottom left are digits that are really ‘0’ and are transcribed as ‘0’, bottom right - digits that are really ‘0’ but are transcribed as ‘9’, top left - digits that are really ‘9’ but are transcribed as ‘0’, …

Top right - Mean probability of correctness for digits from the test dataset, partitioned by location in the image.

Bottom right -  the most-likely digit in each location for a single test case . Digits in blue are correct, in red mistakes.

|

Code to make the figure
-----------------------

Script to make an individual frame - takes epoch as a command-line option:

.. literalinclude:: ../../../models/ATB2_retuned/progress_video/make_frame.py

To make the video, it is necessary to run the script above hundreds of times - giving an image that changes as the training progresses. This script makes the list of commands needed to make all the images, which can be run `in parallel <http://brohan.org/offline_assimilation/tools/parallel.html>`_.

.. literalinclude:: ../../../models/ATB2_retuned/progress_video/make_all_frames.py

To turn the thousands of images into a movie, use `ffmpeg <http://www.ffmpeg.org>`_

.. code-block:: shell

    ffmpeg -r 24 -pattern_type glob -i video/\*.png \
           -c:v libx264 -threads 16 -preset slow -tune film \
           -profile:v high -level 4.2 -pix_fmt yuv420p \
           -b:v 5M -maxrate 5M -bufsize 20M \
           -c:a training.mp4
