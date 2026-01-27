from setuptools import setup, find_packages

setup(
    name="Flow_Perturbation",          
    version="0.1.0",                   
    description="Python package for Flow Perturbation methods",
    packages=find_packages(),           
    python_requires='>=3.8',           
    install_requires=[
        "numpy",
        "torch",
        "PyYAML"
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
