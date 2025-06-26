from setuptools import setup, find_packages

setup(
    name='openml',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy'
    ],
    author='OpenML Contributors',
    description='Common neural network algorithms in PyTorch and NumPy',
    url='https://github.com/yourusername/openml',
)
