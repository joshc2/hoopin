from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]
    
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name = 'hoopin',
    version = '0.1.1',
    description = 'This package is used to analyze NBA team statistics correlated with winning.',
    author = 'Simon Li, Griffen Coburn, Joshua Cabrera',
    author_email = 'joshc2@byu.edu',
    packages = find_packages(),
    install_requires = parse_requirements('requirements.txt'),
    package_data = {'hoopin': ['datasets/*.csv']},
    long_description = long_description
)
