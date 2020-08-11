from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup(
    name='swarm-cg',
    version='v1.0.8',
    description='Tools for automatic parametrization of bonded terms in coarse-grained molecular models, with respect to an all-atom trajectory',
	author='Charly Empereur-mot',
	author_email='charly.empereur@gmail.com',
    license='MIT',
	url = 'https://github.com/GMPavanLab/Swarm-CG',
    long_description=README,
    long_description_content_type="text/markdown",
    keywords = ['gromacs', 'coarse-grain', 'molecular model', 'optimization', 'force field'],
    packages=find_packages(include=['swarmcg', 'swarmcg.*']),
    install_requires=[
        'numpy>=1.16.4',
        'scipy>=1.2.2',
        'pyemd>=0.5.1',
        'matplotlib>=2.2.4',
        'fst-pso>=1.4.12',
        'MDAnalysis>=1.0.0'
    ],
    entry_points={
        'console_scripts': [
        	'scg_optimize=swarmcg.optimize_model:main',
        	'scg_evaluate=swarmcg.evaluate_model:main',
        	'scg_monitor=swarmcg.analyze_optimization:main'
        ]
    }
)



