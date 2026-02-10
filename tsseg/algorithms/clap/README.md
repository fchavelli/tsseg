# ClaSP & CLaP

This directory contains the implementations of the change point and state detection algorithms integrated into the `tsseg` library.

## ClaSP (Classification Score Profile)

**Type:** Change Point Detection

**Description:**
ClaSP is a parameter-free time series segmentation algorithm. It works by calculating a classification score profile (ClaSP) for each time point. Peaks in this profile indicate potential change points. The algorithm uses a validation techniques to identify the most significant change points.

**Main Reference:**
Ermshaus, A., Schäfer, P., & Leser, U. (2023). *ClaSP: Parameter-free time series segmentation*. Data Mining and Knowledge Discovery, 37, 1262–1300.

## CLaP (Classification Label Profile)

**Type:** State Detection

**Description:**
CLaP is an algorithm designed for detecting states or regimes in time series. It relies on ClaSP to identify potential change points, then uses time series classifiers (from the `aeon` library) to assign a state label to each segment.

**Main Reference:**
The CLaP algorithm is based on the work of its authors, building on the foundations of ClaSP. A citation will be added once the paper is accepted.
