from setuptools import setup, find_packages

setup(name='clusterX',
      version='1.0.0.dev5',
      description='Cluster expansion for large-parent-cell materials',
      url='http://sol.physik.hu-berlin.de/cell/',
      author='CELL Developers',
      author_email='srigamonti@physik.hu-berlin.de',
      license='THE-LICENSE',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'sklearn', 'matplotlib','ase','plac','pytest','spglib','sympy','nglview','ipywidgets'],
      entry_points={'console_scripts': ['cell=clusterx.cli.main:main']}
)
