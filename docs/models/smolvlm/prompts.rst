Prompts used to control the extraction
======================================

The same pair of prompts are used for all models.

By convention, there are two prompts:

1. The system prompt, which sets the overall task and context
2. The user prompt, which provides the specific instructions and input data

I've done a bit of experimentation with the prompts, but not much. A major challenge is getting the models to output valid JSON in exactly the format specified. There is probably room for improvement with these prompts, but they work well enough.

.. literalinclude:: ../../../models/smolvlm/prompts.py
