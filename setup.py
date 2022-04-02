from distutils.core import setup
setup(
    name='tinydl',
    packages=['tinydl'],
    version='0.3.0',
    license='MIT',
    description='Python library which facilitates training and validation of deep learning models implemented with PyTorch.',
    author='Michael Jeremias',
    author_email='michael.jeremias.home@gmail.com',
    url='https://github.com/michi-jeremias/tinydl',
    download_url='https://github.com/michi-jeremias/tinydl/archive/refs/tags/v0.2.0.tar.gz',
    keywords=['deep', 'learning', 'pytorch', 'training'],
    install_requires=[
        'torch',
        'sklearn',
        'tqdm',
        'tensorboard'
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        "License :: OSI Approved :: MIT License",   # Again, pick a license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
