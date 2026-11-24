# Evaluation report

## Main table (AbsRel, evaluated on valid GT mask; no per-image scaling)

| System | n | AbsRel mean | 95% CI | by dataset |
|---|---|---|---|---|
| A_calib_B_fallback | 55 | 0.1889 | [0.1468, 0.2368] | eth3d=0.1781(n=17), nyuv2=0.1937(n=38) |
| A_calib_affine | 55 | 0.6245 | [0.5781, 0.6683] | eth3d=0.7368(n=17), nyuv2=0.5742(n=38) |
| A_raw_unscaled | 0 | nan | [nan, nan] |  |
| B_calib_conservative | 55 | 0.1103 | [0.0979, 0.1235] | eth3d=0.1332(n=17), nyuv2=0.1001(n=38) |
| B_calib_consistent | 55 | 0.1916 | [0.1490, 0.2400] | eth3d=0.1901(n=17), nyuv2=0.1922(n=38) |
| B_calib_plain | 55 | 0.2399 | [0.1890, 0.2937] | eth3d=0.2779(n=17), nyuv2=0.2229(n=38) |
| B_metric_raw | 55 | 0.1040 | [0.0943, 0.1141] | eth3d=0.1162(n=17), nyuv2=0.0986(n=38) |
| B_oracle_scale_absrel | 55 | 0.0519 | [0.0437, 0.0616] | eth3d=0.0229(n=17), nyuv2=0.0649(n=38) |
| B_oracle_scale_ls | 55 | 0.0616 | [0.0480, 0.0814] | eth3d=0.0240(n=17), nyuv2=0.0784(n=38) |
| B_unconverted | 0 | nan | [nan, nan] |  |
| submission_missing | 0 | nan | [nan, nan] |  |

## Coverage and solver outcomes

| System | rows | fallback | status counts |
|---|---|---|---|
| A_calib_B_fallback | 55 | 52 | mixed:55 |
| A_calib_affine | 55 | 52 | insufficient:52, ok:3 |
| A_raw_unscaled | 55 | 0 | :55 |
| B_calib_conservative | 55 | 28 | conflict:17, insufficient:11, ok:4, single_anchor:23 |
| B_calib_consistent | 55 | 28 | conflict:17, insufficient:11, ok:4, single_anchor:23 |
| B_calib_plain | 55 | 28 | conflict:17, insufficient:11, ok:4, single_anchor:23 |
| B_metric_raw | 55 | 28 | conflict:17, insufficient:11, ok:4, single_anchor:23 |
| B_oracle_scale_absrel | 55 | 28 | conflict:17, insufficient:11, ok:4, single_anchor:23 |
| B_oracle_scale_ls | 55 | 28 | conflict:17, insufficient:11, ok:4, single_anchor:23 |
| B_unconverted | 55 | 0 | :55 |
| submission_missing | 5 | 0 | unknown:5 |
