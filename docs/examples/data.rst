Data and reproduction
=====================

Running the examples
--------------------

The examples live in the ``examples/gallery`` folder of the source repository;
they are not installed with the ``tsseg`` package. From a checkout:

.. code-block:: bash

   python -m pip install -e '.[aeon,accelerators]' matplotlib
   python -m examples.gallery.jekyll          # one example
   python -m examples.gallery.run_all --output /tmp/tsseg-gallery
   python -m examples.gallery.kmeans_baseline --output /tmp/kmeans.json

Everything runs offline in under a minute. Each example reads a small numeric
snapshot from ``examples/gallery/data`` and checks its SHA-256 hash before use;
the motion-capture example uses the MoCap dataset bundled with tsseg. Only the electricity example needs the ``aeon`` and ``accelerators`` extras (CLaP uses numba).
Without ``--output``, ``run_all`` rewrites the figures, tables and records used
by this documentation; building the documentation itself fits nothing.

Every example follows the same conventions:

* features are standardised with their mean and standard deviation over the
  whole sequence, and the reference columns are never passed to the detector;
* one detector answers each question, with the settings shown on the page;
* the record of each example (downloadable as JSON) holds the features, the
  scaling, the detector parameters, the complete prediction, the snapshot hash
  and the versions of the packages used.

Rebuilding the snapshots
------------------------

The snapshots are derived from public sources, listed below. Rebuilding them
needs network access, and ``librosa`` and ``pyedflib`` for the audio and sleep
recordings:

.. code-block:: bash

   python -m examples.gallery.prepare --cache /tmp/tsseg-gallery/raw
   python -m examples.gallery.prepare --only tides --cache /tmp/tsseg-gallery/raw

Downloads are checked against the hashes in
:download:`sources.json <../../examples/gallery/sources.json>`; a changed
source stops the preparation. OpenAlex counts change as the database is
updated, so they are recorded without a hash. The hashes of the resulting
snapshots are in :download:`manifest.json <../../examples/gallery/data/manifest.json>`.
Books, genomes, recordings and raw responses stay in the cache and are not
redistributed.

.. include:: ../../examples/gallery/data/README.md
   :parser: myst_parser.sphinx_
