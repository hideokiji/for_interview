from setuptools import setup 

with open('requirements.txt') as f:
    required_packages = [ln.strip() for ln in f.readlines()]

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name='for_interview',
    version='0.1',
    description=readme(),

    author='hideokiji',
    license='MIT',
    url='https://github.com/hideokiji/for_interview.git',
    
    package=['for_interview'],
    python_requires=">=3.6",
)
