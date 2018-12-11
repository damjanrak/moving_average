from setuptools import setup, find_packages


# def readme():
#     with open('README.rst') as f:
#         return f.read()


setup(
    name='moving_average',
    version='0.0.1',
    description="implementation of moving average filter using pygears",
    # long_description=readme(),
    packages=find_packages(exclude=['doc']),
    author='Damjan Rakanovic',
    author_email='damjan.rakanovic@gmail.com',
    license='MIT',
    python_requires='>=3.6.0',
    package_data={'': ['*.j2', '*.sv']},
    include_package_data=True,
)
