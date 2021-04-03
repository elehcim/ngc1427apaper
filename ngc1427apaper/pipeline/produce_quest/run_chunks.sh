for i in `seq 1 19`; do
	python morphological_quest_script_multi_iso_hi.py --cache-file=../cache_with_iso.pkl $i
done