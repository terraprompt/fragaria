from setuptools import setup, find_packages

setup(
    name="fragaria",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openai>=1.45.0",
        "aiohttp>=3.10.5",
        "fastapi>=0.114.2",
        "pyyaml>=6.0.2",
        "pydantic>=2.9.1",
        "uvicorn>=0.30.6",
        "aiofiles>=24.1.0",
    ],
    entry_points={
        "console_scripts": [
            "fragaria=fragaria.cli:main",
            "fragaria-server=fragaria.cli:server",
        ],
    },
    author="Dipankar Sarkar",
    author_email="me@dipankar.name",
    description="Advanced Chain of Thought (CoT) Reasoning API with Reinforcement Learning (RL)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/terraprompt/fragaria",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
)