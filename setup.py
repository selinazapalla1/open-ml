from setuptools import setup, find_packages

setup(
    name='open-ml',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy'
    ],
    author='OpenML Contributors',
    description='Common neural network algorithms in PyTorch and NumPy',
    url='https://github.com/selinazapalla1/open-ml',
)
