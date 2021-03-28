
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="soad",
    version="1.0.0",
    author="M.Kıyami ERDİM",
    author_email="kiyami_erdim@outlook.com",

    description="Statistics of Asymmetric Distributions",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/kiyami/soad",

    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',

    install_requires=[
        "matplotlib",
        "scipy",
        "numpy",
    ],
)
