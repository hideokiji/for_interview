from setuptools import setup 

with open('requirements.txt') as f:
    required_packages = [ln.strip() for ln in f.readlines()]

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name='recsys',
    version='0.1',
    description='review pytorch-mlflow',
    long_description=readme(),

    author='hideokiji',
    license='MIT',
    url='https://github.com/hideokiji/for_interview.git',
    
    packages=['recsys'],
    python_requires=">=3.6",
    install_requires=[required_packages],

    entry_points = {
        'console_scripts':[
            'recsys = app.cli:app',
        ],
    }
)
