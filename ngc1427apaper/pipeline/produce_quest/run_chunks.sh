for i in `seq 0 38`; do
	python morphological_quest_script_multi_iso_hi.py --cache-file=../cache_with_iso.pkl $i -s 2000
done