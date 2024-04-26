from setuptools import setup, find_packages

setup(
    name='ZMPY_TF',
    version='0.0.1',
    author='Jhih Siang (Sean) Lai',
    author_email='js.lai@uqconnect.edu.au, jsl035@ucsd.edu',
    description='ZMPY Tensorflow version',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tawssie/ZMPY_TF',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],

    entry_points={
        'console_scripts': [
            'ZMPY_TF_CLI_ZM=ZMPY_TF.ZMPY_TF_CLI_ZM:main',
            'ZMPY_TF_CLI_SuperA2B=ZMPY_TF.ZMPY_TF_CLI_SuperA2B:main',
            'ZMPY_TF_CLI_ShapeScore=ZMPY_TF.ZMPY_TF_CLI_ShapeScore:main',
            'ZMPY_TF_CLI_BatchSuperA2B=ZMPY_TF.ZMPY_TF_CLI_BatchSuperA2B:main',
            'ZMPY_TF_CLI_BatchShapeScore=ZMPY_TF.ZMPY_TF_CLI_BatchShapeScore:main',
            'ZMPY_TF_CLI_BatchZM=ZMPY_TF.ZMPY_TF_CLI_BatchZM:main',
        ],
    },

    python_requires='>=3.9.16',
    install_requires=[
        'numpy>=1.23.5',
        'tensorflow>=2.12.0',
        'tensorflow-probability>=0.20.1'
    ],

    include_package_data=True, # for the cache
)


