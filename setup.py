import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hatespeech-detector",
    version="1.1",
    author="DW Shin",
    author_email="dwshin@idsinternational.com",
    description="Tool for social media hate speech detection",
    long_description=long_description,
    long_description_content_type="text",
    url=None,
    install_requires=['torch==1.7.1+cu101', 'transformers==4.6.1', 'numpy>=1.18.5', 'demoji>=1.1.0'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data = True,
    package_data = {
        'spamdetect' : ['checkpoint/*'], 
    }
)
