from setuptools import setup, find_packages

setup(
    name='py_face_detection',
    version='0.0.2',
    description="Face detection using MTCNN and Face-Net",
    url='https://github.com/uniquetrij/py-face-detection',
    author='Trijeet Modak',
    author_email='uniquetrij@gmail.com',
    packages=find_packages(),
    include_package_data = True,
    zip_safe=False
)
