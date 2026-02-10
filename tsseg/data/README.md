# CMU Graphics Lab Motion Capture Database

This directory contains data derived from the [CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/).

The full dataset contains 2605 trials from more than 140 subjects, organized into 6 main categories and 23 subcategories. The main categories are:

-   **Human Interaction**: Motions involving two subjects.
-   **Interaction with Environment**: Motions on playgrounds, uneven terrain, etc.
-   **Locomotion**: Running, walking, etc.
-   **Physical Activities & Sports**: Basketball, dance, etc.
-   **Situations & Scenarios**: Common behaviors, expressions, pantomime, etc.
-   **Test Motions**

## Data Subset

As labeled data does not seem to be available on the website, this project uses a specific subset from **subject 86** (from sports and various activities). This subset consists of 9 time series: `01`, `02`, `03`, `07`, `08`, `09`, `10`, `11`, and `14`.

## Labeling

The labels for this subset were hard-coded in the source code of Time2State (Wang et al., 2023). An attempt was made to label the remaining time series from this subject, but they appeared to contain errors, with some variables not behaving as expected. This is likely the reason they were originally excluded.

## Preprocessing

For each body part, the original data provides 3 axes of rotation. In this subset, only the first axis was kept for the left and right humerus (arms) and femur (legs).

## License and Terms of Use

The data is provided under a custom permissive license with an attribution requirement. The specific terms are as follows:

-   This data is free for use in research projects.
-   You may include this data in commercially-sold products, but you may not resell this data directly, even in converted form.
-   If you publish results obtained using this data, the creators request that you send the citation of your published paper to `jkh+mocap@cs.cmu.edu` and add the following text to your acknowledgments section:
    > The data used in this project was obtained from mocap.cs.cmu.edu.
    > The database was created with funding from NSF EIA-0196217.

**Source:** Carnegie Mellon University - CMU Graphics Lab - motion capture library.

## Usage in previous work

This dataset was first used for the state detection algorithm in the AutoPlait paper by Matsubara et al. (2014). The authors state in their paper:
> We normalized the values of each dataset so that they had the same mean and variance (i.e., z-normalization).
> The dataset was obtained from the CMU motion capture database. In this dataset, every motion is represented as a sequence of hundreds of frames. It consists of sequences of 64-dimensional vectors, and we chose four dimensions (left-right legs and arms).

This dataset was further used under the same protocol by the authors of Time2State (Wang et al., 2023), who stated:
> This dataset is from the CMU motion capture dataset. Following the setting in [AutoPlait], we choose four dimensions corresponding to left/right arms and legs.

The E2USD paper by Lai et al. (2024) also used this dataset under the same protocol.

## References

1.  **AutoPlait**: Matsubara, Y., Sakurai, Y., and Faloutsos, C. 2014. AutoPlait: Automatic Mining of Co-Evolving Time Sequences. In *Proceedings of the 2014 ACM SIGMOD International Conference on Management of Data* (SIGMOD ’14). Association for Computing Machinery, New York, NY, USA, 193–204. https://doi.org/10.1145/2588555.2588556

2.  **Time2State**: Wang, C., Wu, K., Zhou, T., and Cai, Z. 2023. Time2State: An Unsupervised Framework for Inferring the Latent States in Time Series Data. *Proceedings of the ACM on Management of Data*. 1, 1, Article 1 (May 2023), 18 pages. https://doi.org/10.1145/3589223

3.  **E2USD**: Lai, Z., Li, H., Zhang, D., Zhao, Y., Qian, W., and Jensen, C. S. 2024. E2Usd: Efficient-yet-effective Unsupervised State Detection for Multivariate Time Series. In *Proceedings of the ACM Web Conference 2024*. Association for Computing Machinery, New York, NY, USA, 3010–3021. https://doi.org/10.1145/3589334.3645620