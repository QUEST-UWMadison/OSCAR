from setuptools import setup, find_packages

setup(
    name='oscar',
    version='0.1',
    description='OSCAR: cOmpressed Sensing based Cost lAndscape Reconstruction',
    author='OSCAR Developers',
    url='https://github.com/haoty/QSCAR',
    license='MIT',
    packages=find_packages(exclude=['test*', 'scripts', 'assets', 'notebooks', 'doc']),
    python_requires='>=3.8.13',
    install_requires=[
        'numpy>=1.23.5',
        'scipy>=1.10.1',
        'qiskit>=0.43.0',
        'qiskit_finance>=0.3.4',
        'qiskit_optimization>=0.5.0',
        'qiskit_nature>=0.6.1',
        'networkx>=2.8.8',
        'matplotlib>=3.7.1',
        'cvxpy>=1.3.1',
        'pyscf>=2.2.1'
    ],
    # test_suite='test.test',
)
