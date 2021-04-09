
ROOT=figures
mkdir -p $ROOT
pushd $ROOT
python ../plot_02_good_traj.py
python ../plot_09_r_rps.py
python ../plot_10_m_hi.py
python ../plot_08_panel68p100.py
popd


DIR1=$ROOT/histo
mkdir -p $DIR1
pushd $DIR1
python ../../plot_03_histograms_v2.py
python ../../plot_04_histograms_iso_comparison_with_asym_v2.py
python ../../plot_05_histograms_multi_tol_dist_v2.py
python ../../plot_06_histograms_multi_tol_vel_angle_v2.py
python ../../plot_07_hist2d_asym_kde.py
popd

DIR2=$ROOT/NGC_LIKE
mkdir -p $DIR2
pushd $DIR2
python ../../plot_01_ngc_like.py
popd
