# V2.9.22 OOF subgroup stability and model diversity

- Rows: 699,635
- Training performed: no
- Kaggle submission: no
- Model weights searched: no

## Overall OOF AUC

| Model | AUC | OOF SHA-256 |
|---|---:|---|
| V2915 RealMLP champion | 0.96060130 | 46ad63f08537cd8d61c4c8e8698efc20bb3c8fbad4eb0d94ecd7c993b2d894b9 |
| V2918 LightGBM | 0.96023226 | 409b840fd91375b280d408b80db3841ef884e1d7838adfd205f6160e66a71b52 |
| V2919R2 TabM | 0.95929791 | 4401b44d0fd01a495cf01a4614b7575eef52c8b8144a48b37eb3287a0c139c5d |
| V2920 XGBoost | 0.96001777 | 3cb1f62b3f2250e92a3c07799db193157d8c2be7bd2b3c49d4c7fec7a7e2b39d |
| V2921 CatBoost | 0.95948655 | 2d524fd3fc96ba8bef0024c2c28a0c4cf1ff50e925ff5e519ea4e11e0225fa22 |

## Pairwise Spearman rank correlation

| Model pair | Correlation |
|---|---:|
| V2915 RealMLP champion <> V2918 LightGBM | 0.956289 |
| V2915 RealMLP champion <> V2919R2 TabM | 0.941708 |
| V2915 RealMLP champion <> V2920 XGBoost | 0.954426 |
| V2915 RealMLP champion <> V2921 CatBoost | 0.942427 |
| V2918 LightGBM <> V2919R2 TabM | 0.964842 |
| V2918 LightGBM <> V2920 XGBoost | 0.988169 |
| V2918 LightGBM <> V2921 CatBoost | 0.970234 |
| V2919R2 TabM <> V2920 XGBoost | 0.962079 |
| V2919R2 TabM <> V2921 CatBoost | 0.963732 |
| V2920 XGBoost <> V2921 CatBoost | 0.973568 |

## V2915 RealMLP champion — subgroup AUC

| Group field | Value | Rows | Positive rate | AUC |
|---|---|---:|---:|---:|
| Class | Business | 342212 | 0.7253 | 0.940315 |
| Class | Eco | 327404 | 0.1676 | 0.903984 |
| Class | Eco Plus | 30019 | 0.2422 | 0.929926 |
| Type of Travel | Business travel | 497441 | 0.5843 | 0.957428 |
| Type of Travel | Personal Travel | 202194 | 0.0973 | 0.830127 |
| Customer Type | Loyal Customer | 576990 | 0.4951 | 0.961069 |
| Customer Type | disloyal Customer | 122645 | 0.2009 | 0.923966 |
| Gender | Female | 347952 | 0.4380 | 0.961115 |
| Gender | Male | 351683 | 0.4491 | 0.960058 |
| Flight Distance quintile | (1189.0, 2240.0] | 139817 | 0.5797 | 0.957786 |
| Flight Distance quintile | (2240.0, 4983.0] | 139864 | 0.7503 | 0.942444 |
| Flight Distance quintile | (591.0, 842.0] | 138813 | 0.2948 | 0.945996 |
| Flight Distance quintile | (66.999, 591.0] | 141090 | 0.3100 | 0.946723 |
| Flight Distance quintile | (842.0, 1189.0] | 140051 | 0.2835 | 0.945078 |
| Departure Delay in Minutes band | 0 | 635152 | 0.4480 | 0.960234 |
| Departure Delay in Minutes band | 1-15 | 45856 | 0.4282 | 0.963099 |
| Departure Delay in Minutes band | 16-60 | 18274 | 0.3323 | 0.962984 |
| Arrival Delay in Minutes band | 0 | 640707 | 0.4521 | 0.960142 |
| Arrival Delay in Minutes band | 1-15 | 40284 | 0.3722 | 0.963491 |
| Arrival Delay in Minutes band | 16-60 | 18334 | 0.3059 | 0.961229 |
| Age band | 25-34 | 147648 | 0.3423 | 0.954388 |
| Age band | 35-44 | 172082 | 0.5116 | 0.959348 |
| Age band | 45-54 | 156236 | 0.6014 | 0.956656 |
| Age band | 55+ | 104682 | 0.4659 | 0.964430 |
| Age band | <=24 | 118987 | 0.2439 | 0.936226 |

## V2918 LightGBM — subgroup AUC

| Group field | Value | Rows | Positive rate | AUC |
|---|---|---:|---:|---:|
| Class | Business | 342212 | 0.7253 | 0.939416 |
| Class | Eco | 327404 | 0.1676 | 0.904006 |
| Class | Eco Plus | 30019 | 0.2422 | 0.930883 |
| Type of Travel | Business travel | 497441 | 0.5843 | 0.956851 |
| Type of Travel | Personal Travel | 202194 | 0.0973 | 0.830472 |
| Customer Type | Loyal Customer | 576990 | 0.4951 | 0.960645 |
| Customer Type | disloyal Customer | 122645 | 0.2009 | 0.923834 |
| Gender | Female | 347952 | 0.4380 | 0.960771 |
| Gender | Male | 351683 | 0.4491 | 0.959665 |
| Flight Distance quintile | (1189.0, 2240.0] | 139817 | 0.5797 | 0.957457 |
| Flight Distance quintile | (2240.0, 4983.0] | 139864 | 0.7503 | 0.941017 |
| Flight Distance quintile | (591.0, 842.0] | 138813 | 0.2948 | 0.945701 |
| Flight Distance quintile | (66.999, 591.0] | 141090 | 0.3100 | 0.946385 |
| Flight Distance quintile | (842.0, 1189.0] | 140051 | 0.2835 | 0.945099 |
| Departure Delay in Minutes band | 0 | 635152 | 0.4480 | 0.959879 |
| Departure Delay in Minutes band | 1-15 | 45856 | 0.4282 | 0.962600 |
| Departure Delay in Minutes band | 16-60 | 18274 | 0.3323 | 0.961934 |
| Arrival Delay in Minutes band | 0 | 640707 | 0.4521 | 0.959787 |
| Arrival Delay in Minutes band | 1-15 | 40284 | 0.3722 | 0.962561 |
| Arrival Delay in Minutes band | 16-60 | 18334 | 0.3059 | 0.961210 |
| Age band | 25-34 | 147648 | 0.3423 | 0.953675 |
| Age band | 35-44 | 172082 | 0.5116 | 0.958871 |
| Age band | 45-54 | 156236 | 0.6014 | 0.956281 |
| Age band | 55+ | 104682 | 0.4659 | 0.963943 |
| Age band | <=24 | 118987 | 0.2439 | 0.936605 |

## V2919R2 TabM — subgroup AUC

| Group field | Value | Rows | Positive rate | AUC |
|---|---|---:|---:|---:|
| Class | Business | 342212 | 0.7253 | 0.937510 |
| Class | Eco | 327404 | 0.1676 | 0.902534 |
| Class | Eco Plus | 30019 | 0.2422 | 0.928412 |
| Type of Travel | Business travel | 497441 | 0.5843 | 0.955550 |
| Type of Travel | Personal Travel | 202194 | 0.0973 | 0.828674 |
| Customer Type | Loyal Customer | 576990 | 0.4951 | 0.959895 |
| Customer Type | disloyal Customer | 122645 | 0.2009 | 0.919641 |
| Gender | Female | 347952 | 0.4380 | 0.959963 |
| Gender | Male | 351683 | 0.4491 | 0.958614 |
| Flight Distance quintile | (1189.0, 2240.0] | 139817 | 0.5797 | 0.956970 |
| Flight Distance quintile | (2240.0, 4983.0] | 139864 | 0.7503 | 0.938140 |
| Flight Distance quintile | (591.0, 842.0] | 138813 | 0.2948 | 0.944437 |
| Flight Distance quintile | (66.999, 591.0] | 141090 | 0.3100 | 0.944658 |
| Flight Distance quintile | (842.0, 1189.0] | 140051 | 0.2835 | 0.944275 |
| Departure Delay in Minutes band | 0 | 635152 | 0.4480 | 0.958908 |
| Departure Delay in Minutes band | 1-15 | 45856 | 0.4282 | 0.961988 |
| Departure Delay in Minutes band | 16-60 | 18274 | 0.3323 | 0.961659 |
| Arrival Delay in Minutes band | 0 | 640707 | 0.4521 | 0.958788 |
| Arrival Delay in Minutes band | 1-15 | 40284 | 0.3722 | 0.962395 |
| Arrival Delay in Minutes band | 16-60 | 18334 | 0.3059 | 0.960619 |
| Age band | 25-34 | 147648 | 0.3423 | 0.952591 |
| Age band | 35-44 | 172082 | 0.5116 | 0.957463 |
| Age band | 45-54 | 156236 | 0.6014 | 0.955571 |
| Age band | 55+ | 104682 | 0.4659 | 0.963249 |
| Age band | <=24 | 118987 | 0.2439 | 0.935626 |

## V2920 XGBoost — subgroup AUC

| Group field | Value | Rows | Positive rate | AUC |
|---|---|---:|---:|---:|
| Class | Business | 342212 | 0.7253 | 0.938850 |
| Class | Eco | 327404 | 0.1676 | 0.903950 |
| Class | Eco Plus | 30019 | 0.2422 | 0.930791 |
| Type of Travel | Business travel | 497441 | 0.5843 | 0.956509 |
| Type of Travel | Personal Travel | 202194 | 0.0973 | 0.830554 |
| Customer Type | Loyal Customer | 576990 | 0.4951 | 0.960395 |
| Customer Type | disloyal Customer | 122645 | 0.2009 | 0.923708 |
| Gender | Female | 347952 | 0.4380 | 0.960504 |
| Gender | Male | 351683 | 0.4491 | 0.959505 |
| Flight Distance quintile | (1189.0, 2240.0] | 139817 | 0.5797 | 0.956914 |
| Flight Distance quintile | (2240.0, 4983.0] | 139864 | 0.7503 | 0.940408 |
| Flight Distance quintile | (591.0, 842.0] | 138813 | 0.2948 | 0.945505 |
| Flight Distance quintile | (66.999, 591.0] | 141090 | 0.3100 | 0.946193 |
| Flight Distance quintile | (842.0, 1189.0] | 140051 | 0.2835 | 0.945297 |
| Departure Delay in Minutes band | 0 | 635152 | 0.4480 | 0.959669 |
| Departure Delay in Minutes band | 1-15 | 45856 | 0.4282 | 0.962321 |
| Departure Delay in Minutes band | 16-60 | 18274 | 0.3323 | 0.961871 |
| Arrival Delay in Minutes band | 0 | 640707 | 0.4521 | 0.959547 |
| Arrival Delay in Minutes band | 1-15 | 40284 | 0.3722 | 0.962579 |
| Arrival Delay in Minutes band | 16-60 | 18334 | 0.3059 | 0.961146 |
| Age band | 25-34 | 147648 | 0.3423 | 0.953567 |
| Age band | 35-44 | 172082 | 0.5116 | 0.958409 |
| Age band | 45-54 | 156236 | 0.6014 | 0.955818 |
| Age band | 55+ | 104682 | 0.4659 | 0.963897 |
| Age band | <=24 | 118987 | 0.2439 | 0.936804 |

## V2921 CatBoost — subgroup AUC

| Group field | Value | Rows | Positive rate | AUC |
|---|---|---:|---:|---:|
| Class | Business | 342212 | 0.7253 | 0.937643 |
| Class | Eco | 327404 | 0.1676 | 0.902995 |
| Class | Eco Plus | 30019 | 0.2422 | 0.930066 |
| Type of Travel | Business travel | 497441 | 0.5843 | 0.955825 |
| Type of Travel | Personal Travel | 202194 | 0.0973 | 0.828487 |
| Customer Type | Loyal Customer | 576990 | 0.4951 | 0.959754 |
| Customer Type | disloyal Customer | 122645 | 0.2009 | 0.923227 |
| Gender | Female | 347952 | 0.4380 | 0.960105 |
| Gender | Male | 351683 | 0.4491 | 0.958845 |
| Flight Distance quintile | (1189.0, 2240.0] | 139817 | 0.5797 | 0.956364 |
| Flight Distance quintile | (2240.0, 4983.0] | 139864 | 0.7503 | 0.937987 |
| Flight Distance quintile | (591.0, 842.0] | 138813 | 0.2948 | 0.945130 |
| Flight Distance quintile | (66.999, 591.0] | 141090 | 0.3100 | 0.945921 |
| Flight Distance quintile | (842.0, 1189.0] | 140051 | 0.2835 | 0.944663 |
| Departure Delay in Minutes band | 0 | 635152 | 0.4480 | 0.959112 |
| Departure Delay in Minutes band | 1-15 | 45856 | 0.4282 | 0.962184 |
| Departure Delay in Minutes band | 16-60 | 18274 | 0.3323 | 0.961496 |
| Arrival Delay in Minutes band | 0 | 640707 | 0.4521 | 0.958975 |
| Arrival Delay in Minutes band | 1-15 | 40284 | 0.3722 | 0.962654 |
| Arrival Delay in Minutes band | 16-60 | 18334 | 0.3059 | 0.960721 |
| Age band | 25-34 | 147648 | 0.3423 | 0.953061 |
| Age band | 35-44 | 172082 | 0.5116 | 0.957847 |
| Age band | 45-54 | 156236 | 0.6014 | 0.955155 |
| Age band | 55+ | 104682 | 0.4659 | 0.963179 |
| Age band | <=24 | 118987 | 0.2439 | 0.935877 |

## Interpretation

Subgroup AUCs show where each model ranks passengers well or poorly. Correlation reveals overlap among model rankings. Neither metric alone justifies promotion; any blend must be re-evaluated on the same frozen OOF and fold splits.
