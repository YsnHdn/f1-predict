"""
Setup configuration for the F1 prediction project.
"""

from setuptools import setup, find_packages

setup(
    name="f1-predict",
    version="0.1.0",
    description="Formula 1 prediction system using FastF1 and CrewAI",
    author="Yassine Handane",
    author_email="y.handane@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastf1",
        "pandas",
        "numpy",
        "matplotlib",
        "crewai",
        "python-dotenv",
        "streamlit",
        "scikit-learn",
        "pytest",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "black",
        ],
    },
    python_requires=">=3.8",
)