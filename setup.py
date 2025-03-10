from setuptools import setup, find_packages

setup(
    name='vex-high-stakes-gym',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='OpenAI Gym environment for the VEX game "High Stakes".',
    packages=find_packages(),
    install_requires=[
        'gym',
        'numpy',
        'matplotlib',
        'pygame'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)