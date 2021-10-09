import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    long_description = f.read()

setup(
    name="mi-module-zoo",
    packages=find_packages(),
    license="MIT",
    package_dir={"mi-module-zoo": "mi_module_zoo"},
    test_suite="tests",
    python_requires=">=3.6.1",
    description="Reusable PyTorch Modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Machine Intelligence",
    package_data={"ptgnn": ["py.typed"]},
    install_requires=[
        "torch>=1.9.1",
        "typing-extensions",
    ],
    extras_require={"dev": ["black", "isort", "pre-commit", "numpy"], "aml": ["azureml"]},
    setup_requires=["setuptools_scm"],
    url="https://github.com/microsoft/mi-module-zoo/",
    project_urls={
        "Bug Tracker": "https://github.com/microsoft/mi-module-zoo/issues",
        # "Documentation": "https://github.com/microsoft/mi-module-zoo/tree/master/docs",
        "Source Code": "https://github.com/microsoft/mi-module-zoo",
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Typing :: Typed",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    zip_safe=False,
)
