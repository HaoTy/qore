from setuptools import setup, find_packages

setup(
    name='qore',
    version='0.1',
    description='Q Ore leverages quantum power to speedup open pit mining algorithms',
    author='Q Ore Team',
    url='https://github.com/cs210/IBM-QC-Open-Pit-Mining/',
    packages=find_packages(exclude=['test*', 'scripts', 'assets']),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.17',
        'qiskit>=0.25',
        'networkx>=2.5.1',
        'prettytable>=2.1.0',
        'pseudoflow>=1.15.0'
    ],
    # test_suite='test.test',
)
