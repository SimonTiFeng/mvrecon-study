# Correspondence sweep (HPC2)

records: 3752

## Accuracy and cost (no injected outliers)

| matcher | mask | solve | n | rot med | rot<2° | lat med | lat<0.5m | cost med (ms) |
|---|---|---|---|---|---|---|---|---|
| akaze | none | corrected | 40 | 3.72° | 12 | 0.63 m | 18 | 350 |
| akaze | none | rot_fixed | 40 | 5.02° | 14 | 0.47 m | 23 | 346 |
| akaze | sky | corrected | 40 | 3.71° | 12 | 0.61 m | 20 | 350 |
| akaze | sky | rot_fixed | 40 | 5.02° | 14 | 0.47 m | 23 | 345 |
| akaze | sky_far | corrected | 40 | 3.71° | 9 | 0.61 m | 20 | 350 |
| akaze | sky_far | rot_fixed | 40 | 5.02° | 14 | 0.46 m | 23 | 345 |
| orb | none | corrected | 37 | 3.75° | 12 | 0.71 m | 16 | 155 |
| orb | none | rot_fixed | 37 | 3.44° | 14 | 0.49 m | 20 | 149 |
| orb | sky | corrected | 36 | 3.10° | 12 | 0.65 m | 16 | 155 |
| orb | sky | rot_fixed | 36 | 3.44° | 14 | 0.46 m | 22 | 149 |
| orb | sky_far | corrected | 36 | 3.10° | 12 | 0.65 m | 16 | 155 |
| orb | sky_far | rot_fixed | 36 | 3.44° | 14 | 0.46 m | 22 | 149 |
| sift | none | corrected | 40 | 2.12° | 16 | 0.51 m | 20 | 822 |
| sift | none | rot_fixed | 40 | 5.02° | 14 | 0.47 m | 22 | 816 |
| sift | sky | corrected | 40 | 2.12° | 16 | 0.50 m | 20 | 822 |
| sift | sky | rot_fixed | 40 | 5.02° | 14 | 0.47 m | 22 | 816 |
| sift | sky_far | corrected | 40 | 2.12° | 16 | 0.47 m | 20 | 822 |
| sift | sky_far | rot_fixed | 40 | 5.02° | 14 | 0.46 m | 22 | 816 |
| sift_akaze | none | corrected | 40 | 2.33° | 13 | 0.53 m | 17 | 1131 |
| sift_akaze | none | rot_fixed | 40 | 5.02° | 14 | 0.47 m | 23 | 1125 |
| sift_akaze | sky | corrected | 40 | 2.32° | 13 | 0.53 m | 17 | 1131 |
| sift_akaze | sky | rot_fixed | 40 | 5.02° | 14 | 0.47 m | 23 | 1125 |
| sift_akaze | sky_far | corrected | 40 | 2.32° | 10 | 0.53 m | 17 | 1130 |
| sift_akaze | sky_far | rot_fixed | 40 | 5.02° | 14 | 0.46 m | 23 | 1125 |

## Pareto front (cost vs median lateral error)

| matcher | mask | solve | lat med | cost med (ms) |
|---|---|---|---|---|
| orb | none | rot_fixed | 0.49 m | 149 |
| orb | sky_far | rot_fixed | 0.46 m | 149 |
| akaze | sky_far | rot_fixed | 0.46 m | 345 |
| sift | sky_far | rot_fixed | 0.46 m | 816 |

## Cross-scene transfer — sift / sky_far / rot_fixed

| scene | n | rot med | lat med | lat<0.5m | baseline lat med |
|---|---|---|---|---|---|
| courtyard | 5 | 1.49° | 0.10 m | 5 | 3.31 m |
| delivery_area | 5 | 1.48° | 0.28 m | 3 | 1.60 m |
| electro | 6 | 46.88° | 1.05 m | 3 | 10.81 m |
| facade | 6 | 3.44° | 1.32 m | 2 | 1.08 m |
| office | 6 | 78.65° | 0.35 m | 3 | 1.37 m |
| playground | 6 | 16.45° | 1.04 m | 3 | 3.31 m |
| terrace | 6 | 46.34° | 1.65 m | 3 | 1.92 m |

## Robustness to gross outlier correspondences

| solve | outlier | rot med | lat med | lat<0.5m |
|---|---|---|---|---|
| corrected | 0% | 3.38° | 0.59 m | 217 |
| corrected | 10% | 3.26° | 0.64 m | 224 |
| corrected | 20% | 3.38° | 0.63 m | 225 |
| corrected | 30% | 3.18° | 0.62 m | 210 |
| rot_fixed | 0% | 4.47° | 0.47 m | 268 |
| rot_fixed | 10% | 4.47° | 0.47 m | 267 |
| rot_fixed | 20% | 4.47° | 0.47 m | 265 |
| rot_fixed | 30% | 4.47° | 0.47 m | 262 |
