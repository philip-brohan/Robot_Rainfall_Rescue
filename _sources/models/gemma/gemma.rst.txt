Gemma-3-4b run alone
====================

The Gemma model comes in a range of sizes. here I'm using the 4b version (smallest and fastest version that has vision capabilities). The 12b version would work instead (possibly better). I haven't tried any of the others (larger sizes are too big to train oon one H100).

We can do exactly the same with Gemma as :doc:`we did with SmolVLM <../../index>`.

The code is mostly the same (just change the ```---model``` argument), but the scripts to run Gemma are slightly different.

.. toctree::
   :maxdepth: 1

   Gemma-3-4b run on a single image <extract>
   Prompts used to control the extraction <../smolvlm/prompts>
   Plot extracted vs image data <plot_image+extracted>
   Gemma-3-4b run on a set of images <extract_multi>
   Plot success percentages from a model <plot_stats>
