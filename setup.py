import os
from setuptools import setup
import glob

scripts = glob.glob('ngc1427apaper/plots/plot_*')

setup(name='ngc1427apaper',
      version='0.1.1',
      description='Package for NGC1427A Mastropietro et al 2020 paper',
      license="MIT",
      author='Michele Mastropietro',
      author_email='michele.mastropietro@ugent.be',
      install_requires=['simulation', 'astropy', 'pandas', 'matplotlib', 'scipy', 'numpy',
                        'pynbody', 'streamlit',
                        'scikit-image', 'tqdm', 'photutils'],
      packages=['ngc1427apaper', 'ngc1427apaper.plots'],
      package_data={'ngc1427apaper': ['data/*']},
      scripts=scripts,
      entry_points={}
      )
