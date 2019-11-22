from setuptools import setup
from setuptools import find_packages

setup(name='gravity_gae',
      description='Gravity-Inspired Graph Autoencoders for Directed Link Prediction',
      author='Deezer Research',
      install_requires=['networkx==2.2',
                        'numpy',
                        'scikit-learn',
                        'scipy',
                        'tensorflow==1.*'],
      package_data={'gravity_gae': ['README.md']},
      packages=find_packages())