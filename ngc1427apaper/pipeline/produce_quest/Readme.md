# Scripts to create data files from simulations

* `get_solutions.py` computes the solutions of the galaxy orientation given `rp` and `vp` (and their tolerances).
* `dff.py` creates `cache_with_iso.pkl`
* `morphological_quest_script_multi_iso_hi.py` computes angles between tails for galaxy with solutions using different surface brightness limits. Writes chunks into `quest/iso_hi` folder.
* `merge_data.py` merges `cache_with_iso.pkl` with `quest/iso_hi`.
* `asym.py` Computes asymmetry writing into `quest/morph`.
* `join_morph.py` inserts asym.py output with previous cache with isophote computation and HI data.
* `add_stars_and_others.py` add other simulation data like stellar mass, effective radius, star formation rate, ram pressure.
