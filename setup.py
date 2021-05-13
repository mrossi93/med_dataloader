#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['tensorflow>=2.4', 'SimpleITK>=2.0', "click"]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Matteo Rossi",
    author_email='rossimatteo1993@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A general-purpose Dataloader for Tensorflow 2.x. It supports many medical image formats.",  # noqa
    entry_points={'console_scripts': [
        'med_dataloader=med_dataloader.cli:main']},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='med_dataloader',
    name='med_dataloader',
    packages=find_packages(include=['med_dataloader', 'med_dataloader.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mrossi93/med_dataloader',
    version='0.1.10',
    zip_safe=False,
)
