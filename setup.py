from setuptools import setup, find_packages

description = "Python Machine Learning Demo"

setup(
    name="testpml",
    version="1.0.0",
    description='Python Machine Learning Demo',
    long_description=description,
    url="",
    license="",
    author="Ryan Hobbs",
    author_email="rhobbs@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        # Reference: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers"],
    install_requires=[
        'ipython >= 5.1.0',
        'pandas >= 0.19.2',
        'numpy >= 1.11.3',
        'scikit-learn >= 0.18.1',
        'scipy >= 0.18.1',
        'matplotlab >= 1.5.3'
    ],
    entry_points={
        'console_scripts':['pmldemo=pmldemo:main']},
)