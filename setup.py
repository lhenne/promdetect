import setuptools

with open("README.md", "r") as readmeFile:
    longDescription = readmeFile.read()

setuptools.setup(
    name="promdetect",
    version="0.1dev",
    author="Lukas Henne",
    author_email="lhenne@posteo.de",
    description="An acoustical prominence detector based on Deep Learning",
    long_description=longDescription,
    long_description_content_type="text/markdown",
    url="https://github.com/lhenne/promdetect",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.3'
)