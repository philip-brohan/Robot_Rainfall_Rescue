Models for layout analysis
==========================

For the :doc:`simplified training dataset <../training_data/scripts>` we :doc:`can do transcription directly <atb2>`. For a more realistic training dataset this direct approach is much harder - the data are much more complicated so we need a much more complex model, and this is both hard to train and runs into practical limitations such as using too much GPU RAM.

So it's worth breaking the problem down into sections - what if we first just train a model to find data locations on the page, then we can cut the page up into sections and do transcription on each section separately.

.. toctree::
   :maxdepth: 1

   Finding data locations directly <find_grid/index>
   Finding grid_corners <find_corners/index>



 

