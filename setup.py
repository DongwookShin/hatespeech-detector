from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hateSpeechDetect",
    version="1.0",
    author="DW Shin",
    author_email="dwshin@idsinternational.com",
    description="Tool for social media hate speech detection",
    long_description=long_description,
    long_description_content_type="text",
    url='https://download.pytorch.org/whl/torch_stable.html',
    install_requires=['torch==1.7.1+cu101', 'torchvision==0.8.2+cu101', 'transformers==4.6.1', 'numpy>=1.18.5', 'demoji>=1.1.0'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data = True,
    package_data = {
        'hateSpeechDetect' : ['pretrained/*.*'] 
    }
)
