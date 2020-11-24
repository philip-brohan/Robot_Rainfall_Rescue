Getting the Rainfall Rescue data
================================

The images being used are the 10-year rainfall sheets: lose-leaf forms, each recording monthly rainfall at a single UK station over a period of 10-years. The `collection of these in the UK Meteorological Archive <https://digital.nmla.metoffice.gov.uk/SO_d383374a-91c3-4a7b-ba96-41b81cfb9d67/>`_ comprises about 65,000 such sheets covering 1677-1960 (though the early years include very few stations). 

A key advantage of these documents is that they were manually digitised in spring 2020 by the `Rainfall Rescue citizen science project <https://www.zooniverse.org/projects/edh/rainfall-rescue>`_, and Ed Hawkins, PI of that project, is sharing the transcriptions through `a gitHub repository <https://github.com/ed-hawkins/rainfall-rescue>`_. So we can get image::transcription pairs from that repository.

Download the repository:

.. literalinclude:: ../../rainfall_rescue/get_data/download_Ed_RR_data.sh


Extract the available image::transcription pairs:

.. literalinclude:: ../../rainfall_rescue/get_data/copy_pairs.py


