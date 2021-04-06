# pipeline.sh
python morphological_quest_script.py $sims
python dff.py # creates:cache_with_iso.pkl
python morphological_quest_script_multi_iso_hi.py $chunks
python merge_data.py

python asym.py
python join_morph.py

python add_stars_and_others.py