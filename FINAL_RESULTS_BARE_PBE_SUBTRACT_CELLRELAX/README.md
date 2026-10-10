# FINAL_RESULTS_BARE_PBE_SUBTRACT_CELLRELAX — bare-PBE phonons at the 16 refined cells

Generated 2026-10-10 by `cellrelax/subtract_d3_cellrelax.py` (stage1 / stage2), which runs the unchanged
`phonopy/subtract_d3_force_sets.py` and `phonopy/postprocess_bare_pbe_subtract.py` with the reference redirected to
`FINAL_RESULTS_CELLRELAX/` (ISIF=4-refined PBE+D3 cells, `cellrelax/README.md`) and the output to this directory.
Same definitions, file layout and columns as `FINAL_RESULTS_BARE_PBE_SUBTRACT/README.md`; the other 32 TMDs of the
48-material set stay there (their cells were already at the PBE+D3 equilibrium).

16 materials; hiphive refit applied to 7 (MoS2, WS2_WSe2_3R, MoS2_MoSe2_3R, MoSe2_WS2_2H, WSe2_WTe2_2H, MoSe2_WS2_3R, WSe2_WTe2_3R)
incl. forced on MoS2, MoS2_MoSe2_3R, WSe2_WTe2_2H.
Lowest band-path frequency after post-processing: -0.0000 THz (MoS2).
`rmse_vs_pbed3_THz` compares against `FINAL_RESULTS_CELLRELAX/<mat>/band.yaml` (the refined-cell PBE+D3 reference).
See `summary.csv`.
