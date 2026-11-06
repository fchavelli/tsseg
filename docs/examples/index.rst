Examples
========

Eight worked examples on public data. Each one asks a question about a real
record, answers it with a single tsseg detector, and compares the result with
a reference that the detector never sees: chapter boundaries, dated storms,
gene annotations or expert labels. Figures, data and code can be downloaded,
and every example runs offline from a source checkout.

.. toctree::
   :hidden:
   :maxdepth: 1

   jekyll
   storm_surge
   chloroplast
   climate_vocabulary
   mocap
   electricity
   speakers
   sleep
   data

.. rubric:: Change points and states

tsseg addresses two related tasks. **Change point detection** finds the
positions where the behaviour of a sequence changes. **State detection**
assigns a label to every observation, so that a behaviour that comes back
receives the same label each time. Three periods of electricity generation,
windy, then gas-heavy, then windy again, contain two change points but only
two states:

.. raw:: html

   <div class="tsseg-sequence" role="img" aria-label="Windy, then gas-heavy, then windy again: three periods, two change points, two states.">
     <span>Windy</span><span>Gas-heavy</span><span>Windy again</span>
   </div>

The ordered axis need not be time: two examples follow positions in a text and
along a genome.

.. raw:: html
   :file: _cards.html

.. rubric:: Reading the examples

* The reference shown in each figure (chapters, storm dates, annotations,
  expert labels) is never part of the detector's input. It is only used to
  interpret the output.
* The examples, their features and their settings were chosen after
  inspecting results. The scores describe these records; they are not
  estimates of performance on new data.
* State labels (S0, S1, …) are arbitrary names, ordered by the mean of the
  first plotted feature. Their colours carry no meaning across figures.
* The adjusted Rand index (ARI) compares two labellings: 1 means identical
  partitions, values near 0 mean chance-level agreement.

:doc:`data` explains how to run the examples and lists the sources and their
licences.
