from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(
    name="mlgen",
    version="1.0.4",
    description="MLGen is a tool which helps you to generate machine learning code with ease.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/NebutechOpenSource/MLGen',
    author="Nebutech",
    author_email="mukundh.bhushan@nebutech.in",
    license="MIT",
    classifiers=[ 
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    "License :: OSI Approved :: MIT License",

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
    packages=["mlgen"],
    include_package_data=True,
    install_requires=["pyyaml"],
    entry_points={
        "console_scripts": [
            "mlgen = mlgen.__main__:main",
        ]
    },
)