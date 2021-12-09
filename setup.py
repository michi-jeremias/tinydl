import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='tinydl',
    version='0.1',
    scripts=[],
    author="Michael Jeremias",
    author_email="michael.jeremias.home@gmail.com",
    description="A Python library for deeplearning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/michi-jeremias/tinydl",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3.0",
         "Operating System :: OS Independent",
    ],
)
