from setuptools import setup, find_packages

setup(
    name="cali",
    version="0.1.0",
    packages=find_packages(where="src"),
    description="A method for trust estimation in classification of deep learning models.",
)