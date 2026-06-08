# Grand Analysis

Servers included: 8

This is an across-server descriptive summary. Server-level outputs are summarized first, then combined with metric-specific N weights.

## Key N-Weighted Server Metrics

| metric | n_servers | weight_col | weight_sum | weighted_mean | weighted_sd | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target_best_period | 8 | server_n_records | 146937582.000 | 23.681 | 1.931 | 12.000 | 24.000 |
| win_rate_best_period | 8 | server_n_records | 146937582.000 | 22.553 | 5.433 | 8.000 | 33.000 |
| performance_pc1_explained | 8 | server_n_records | 146937582.000 | 0.631 | 0.054 | 0.430 | 0.675 |
| performance_pc2_explained | 8 | server_n_records | 146937582.000 | 0.192 | 0.026 | 0.156 | 0.256 |
| performance_pc3_explained | 8 | server_n_records | 146937582.000 | 0.087 | 0.011 | 0.078 | 0.138 |
| success_pc1_win_rate_loading | 8 | server_n_records | 146937582.000 | 0.327 | 0.095 | 0.039 | 0.389 |
| success_pc2_win_rate_loading | 8 | server_n_records | 146937582.000 | 0.170 | 0.103 | -0.045 | 0.355 |
| success_pc3_win_rate_loading | 8 | server_n_records | 146937582.000 | 0.326 | 0.218 | 0.125 | 0.966 |
| pc1_phase_fdr_significant | 8 | pc1_phase_players | 8000.000 | 216.750 | 195.742 | 93.000 | 712.000 |
| pc2_phase_fdr_significant | 8 | pc2_phase_players | 8000.000 | 185.000 | 232.593 | 14.000 | 759.000 |
| deltammr_phase_fdr_significant | 8 | deltammr_phase_players | 8000.000 | 3.250 | 1.854 | 1.000 | 6.000 |

## Within-Subject Period Peaks

| metric | n_servers | weight_col | total_valid_players | weighted_best_period | weighted_sd_best_period | weighted_sem_best_period |
| --- | --- | --- | --- | --- | --- | --- |
| PC1 | 7 | valid_players | 7000 | 24.000 | 0.000 | 0.000 |
| PC2 | 7 | valid_players | 7000 | 24.000 | 0.000 | 0.000 |
| DeltaMMR | 7 | valid_players | 7000 | 24.000 | 0.000 | 0.000 |

## FDR Phase Counts

| metric | n_servers | weight_col | total_players_analyzed | total_fdr_significant | weighted_fdr_fraction |
| --- | --- | --- | --- | --- | --- |
| PC1 | 8 | players_analyzed | 8000 | 1734 | 0.217 |
| PC2 | 8 | players_analyzed | 8000 | 1480 | 0.185 |
| DeltaMMR | 8 | players_analyzed | 8000 | 26 | 0.003 |

## Circular Model Preference

| metric | n_servers | weight_col | total_fdr_significant | fit_servers | skipped_servers | preferred_1_component_phases | preferred_2_component_phases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PC1 | 8 | n_fdr_significant | 1734 | 8 | 0 | 845 | 889 |
| PC2 | 8 | n_fdr_significant | 1480 | 7 | 1 | 707 | 759 |
| DeltaMMR | 8 | n_fdr_significant | 26 | 0 | 8 | 0 | 0 |

## Phase-Count Weighted PC Peak Density

| metric | n_servers | total_phases | weighted_peak_hour | kappa | min_phases |
| --- | --- | --- | --- | --- | --- |
| PC1 | 8 | 1734 | 22.200 | 4.000 | 20 |
| PC2 | 7 | 1466 | 10.400 | 4.000 | 20 |

## Pooled Circular Bimodality Test

| metric | n_servers | n_phases | preferred | delta_bic_1_minus_2 | component_1_h | component_2_h | component_1_weight | component_2_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC1 | 8 | 1734 | 2-component | 24.514 | 21.377 | 7.371 | 0.615 | 0.385 |
| PC2 | 7 | 1466 | 2-component | 27.892 | 9.881 | 20.073 | 0.639 | 0.361 |
| DeltaMMR | 0 | 0 |  |  |  |  |  |  |
