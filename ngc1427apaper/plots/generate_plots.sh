SCRIPT=`realpath $0`
DIR=`dirname $SCRIPT`

ROOT=figures
mkdir -p $ROOT
pushd $ROOT
python $DIR/plot_02_good_traj.py
python $DIR/plot_09_r_rps.py
python $DIR/plot_10_m_hi.py
python $DIR/plot_08_panel68p100.py
popd


DIR1=$ROOT/histo
mkdir -p $DIR1
pushd $DIR1
python $DIR/plot_03_histograms_v2.py
python $DIR/plot_04_histograms_iso_comparison_with_asym_v2.py
python $DIR/plot_05_histograms_multi_tol_dist_v2.py
python $DIR/plot_06_histograms_multi_tol_vel_angle_v2.py
python $DIR/plot_07_hist2d_asym_kde.py
popd

DIR2=$ROOT/NGC_LIKE
mkdir -p $DIR2
pushd $DIR2
python $DIR/plot_01_ngc_like.py
popd
