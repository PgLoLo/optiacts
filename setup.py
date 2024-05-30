from setuptools import setup, find_packages


setup(
    name='optiacts',
    version='0.1.0',
    description='Optimized Nonlinear Activation Functions for PyTorch',
    author='PG_LoLo',
    author_email='novgosh@yandex.ru',
    url='https://github.com/PgLoLo/optiacts',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
