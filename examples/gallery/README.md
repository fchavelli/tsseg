# tsseg examples

Eight examples on public data, each answering one question with one tsseg
detector and comparing the result with a reference that the detector never
sees. They are rendered in the [Examples](https://fchavelli.github.io/tsseg/examples/index.html)
section of the documentation.

| Example | Question | Detector | Reference |
|---|---|---|---|
| `jekyll` | Where does *Jekyll and Hyde* switch to first-person testimony? | PELT | Chapter structure |
| `storm_surge` | What do storms add to the tide at The Battery, New York? | PELT | Dated storms (NOAA) |
| `chloroplast` | Which regions of a chloroplast genome are GC-rich? | PELT | rRNA gene annotation |
| `climate_vocabulary` | When did four expressions take hold in climate research titles? | PELT | Context milestones |
| `mocap` | Can a state detector recognise recurring human activities? | TICC (4 states given) | Activity labels |
| `electricity` | Does Britain's generation mix return to an earlier state? | CLaP | Carbon intensity, weather |
| `speakers` | Can a returning voice be recognised? | TICC (3 states given) | Speaker names |
| `sleep` | Do unsupervised states match expert sleep stages? | HDP-HSMM | Expert hypnogram |

```sh
python -m pip install -e '.[aeon,accelerators]' matplotlib
python -m examples.gallery.jekyll
python -m examples.gallery.run_all --output /tmp/tsseg-gallery
python -m examples.gallery.kmeans_baseline --output /tmp/kmeans.json
```

The examples run offline from the hash-checked snapshots in `data/`. Without
`--output`, `run_all` rewrites the figures, tables, records and thumbnails
under `docs/`; Sphinx only reads them. `kmeans_baseline` compares the two
labelled state examples with k-means, which ignores time order.

To rebuild the snapshots from the original sources (network access; `librosa`
and `pyedflib` for the audio and sleep recordings):

```sh
python -m examples.gallery.prepare --cache /tmp/tsseg-gallery/raw
```

Source URLs and hashes are in `sources.json`, snapshot hashes in
`data/manifest.json`, sources and licences in [`data/README.md`](data/README.md).
