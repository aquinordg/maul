from setuptools import setup

with open("README.md", "r") as arq:
    readme = arq.read()

setup(name='maul',
    version='0.0.1',
    license='MIT License',
    author='R. Douglas G. Aquino',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='aquinordga@gmail.com',
    keywords='machine learning',
    description = 'maul: MAin Useful tools in machine Learning',
    packages=['src'],
    install_requires=['pandas','numpy','scikit-learn'])
