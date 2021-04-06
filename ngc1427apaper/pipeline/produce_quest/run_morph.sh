for i in `seq 0 38`; do
	python asym.py --cache-file=../cache_with_multi_iso_and_hi_valid.pkl $i -s 2000
done