from setuptools import setup, find_packages

setup(
    name="mllibraries",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch", "torchvision"],
    python_requires=">=3.7",
)
