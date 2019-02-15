from setuptools import setup, find_packages

setup(name='clusterX',
      version='1.0.0.dev6',
      description='CELL (aka clusterX) is a python package for building Cluster Expansion models of simple and complex alloys and performing thermodynamical analyses of materials.',
      url='http://sol.physik.hu-berlin.de/cell/',
      author='CELL Developers',
      author_email='srigamonti@physik.hu-berlin.de',
      license='http://www.apache.org/licenses/LICENSE-2.0',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'sklearn', 'matplotlib','ase','plac','pytest','spglib','sympy','nglview','ipywidgets'],
      entry_points={'console_scripts': ['cell=clusterx.cli.main:main']}
)
