from setuptools import setup, find_packages

import swarmcg

with open('README.md') as readme_file:
    README = readme_file.read()

setup(
    name='swarm-cg',
    version=swarmcg.__version__,
    description='Tools for automatic parametrization of bonded terms in coarse-grained molecular models, with respect to an all-atom trajectory',
    author='Charly Empereur-mot',
    author_email='charly.empereur@gmail.com',
    license='MIT',
    url='https://github.com/GMPavanLab/Swarm-CG',
    download_url=f'https://github.com/GMPavanLab/Swarm-CG/archive/v{swarmcg.__version__}.tar.gz',
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=['gromacs', 'coarse-grain', 'molecular model', 'optimization', 'force field'],
    packages=find_packages(include=['swarmcg', 'swarmcg.*']),
    install_requires=[
        'numpy>=1.16.4',
        'scipy>=1.2.2',
        'pyemd>=0.5.1',
        'matplotlib>=3.2.2',  # warnings appear for lower versions, don't change
        'fst-pso>=1.4.12',
        'MDAnalysis>=1.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points={
        'console_scripts': [
            'scg_optimize=swarmcg.optimize_model:main',
            'scg_evaluate=swarmcg.evaluate_model:main',
            'scg_monitor=swarmcg.analyze_optimization:main'
        ]
    }
)



