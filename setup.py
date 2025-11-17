from setuptools import setup, find_packages

setup(
    name='cosmos',
    version='0.1.0',
    packages=find_packages(),
    py_modules=['cosmos_run', 'cosmos_search', 'cosmos_utils'],
    entry_points={
        'console_scripts': [
            'cosmos = cosmos_run:main',
        ],
    },
    install_requires=[
        'ase>=3.22.1',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'requests>=2.26.0',
        'python-json-logger>=2.0.1',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='CoSMoS Global Structure Search Package',
    long_description=open('ReadMe.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/cosmos',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)