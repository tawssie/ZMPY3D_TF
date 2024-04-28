from setuptools import setup, find_packages

setup(
    name='ZMPY3D_TF',
    version='0.0.2',
    author='Jhih Siang (Sean) Lai',
    author_email='js.lai@uqconnect.edu.au, jsl035@ucsd.edu',
    description='ZMPY3D Tensorflow version',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tawssie/ZMPY3D_TF',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],

    entry_points={
        'console_scripts': [
            'ZMPY3D_TF_CLI_ZM=ZMPY3D_TF.ZMPY3D_TF_CLI_ZM:main',
            'ZMPY3D_TF_CLI_SuperA2B=ZMPY3D_TF.ZMPY3D_TF_CLI_SuperA2B:main',
            'ZMPY3D_TF_CLI_ShapeScore=ZMPY3D_TF.ZMPY3D_TF_CLI_ShapeScore:main',
            'ZMPY3D_TF_CLI_BatchSuperA2B=ZMPY3D_TF.ZMPY3D_TF_CLI_BatchSuperA2B:main',
            'ZMPY3D_TF_CLI_BatchShapeScore=ZMPY3D_TF.ZMPY3D_TF_CLI_BatchShapeScore:main',
            'ZMPY3D_TF_CLI_BatchZM=ZMPY3D_TF.ZMPY3D_TF_CLI_BatchZM:main',
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


