from distutils.core import setup
setup(
    name='YOURPACKAGENAME',         # How you named your package folder (MyLib)
    packages=['YOURPACKAGENAME'],   # Chose the same as "name"
    version='0.1',      # Start with a small number and increase it with every change you make
    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license='MIT',
    # Give a short description about your library
    description='TYPE YOUR DESCRIPTION HERE',
    author='YOUR NAME',                   # Type in your name
    author_email='your.email@domain.com',      # Type in your E-Mail
    # Provide either the link to your github or to your website
    url='https://github.com/user/reponame',
    # I explain this later on
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',
    # Keywords that define your package best
    keywords=['SOME', 'MEANINGFULL', 'KEYWORDS'],
    install_requires=[            # I get to this in a second
        'validators',
        'beautifulsoup4',
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        # Define that your audience are developers
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        "License :: OSI Approved :: GNU General Public License v3.0",   # Again, pick a license
        # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
# import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()


# setuptools.setup(
#     name='tinydl',
#     version='0.1',
#     scripts=[],
#     author="Michael Jeremias",
#     author_email="michael.jeremias.home@gmail.com",
#     description="A Python library for deeplearning",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/michi-jeremias/tinydl",
#     packages=setuptools.find_packages(),
#     classifiers=[
#          "Programming Language :: Python :: 3",
#          "License :: OSI Approved :: GNU General Public License v3.0",
#          "Operating System :: OS Independent",
#     ],
# )
