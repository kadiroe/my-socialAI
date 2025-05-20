from setuptools import setup, find_packages

setup(
    name="socialAI",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
    ],
    author="Your Name",
    description="AI-powered social media content generation",
    python_requires=">=3.8",
)