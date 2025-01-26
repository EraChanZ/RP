from setuptools import setup, find_packages

setup(
    name="custom_eval_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "Pillow",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for evaluating hand landmark predictions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)