Huggingface interface functions
===============================

Some scripts download models from Huggingface, so they will need to login first. This library file contains a function to do that.
Logging in requires an API key which you can get from the Huggingface web-site when you create an account there.
Store that api key in file ```~/.huggingface_api``` and you can use the ```HFLogin()``` function in this library to connect.

.. literalinclude:: ../../RR_utils/hf.py
