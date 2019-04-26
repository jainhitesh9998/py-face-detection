from setuptools import setup, find_packages
import pip

setup(
    name='py_face_detection',
    version='0.0.1',
    description="Face detection API",
    url='https://github.com/uniquetrij/py-face-detection',
    author='Trijeet Modak',
    author_email='uniquetrij@gmail.com',
    packages=find_packages(),
    include_package_data = True,
    zip_safe=False
)
