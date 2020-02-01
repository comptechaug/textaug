from setuptools import setup, find_packages
from os.path import join, dirname
setup(name='textaug',
      version='0.2',
      description='Little package for text augmentation',
      author='TextAugmentationTeam',
      license='MIT',
      packages=find_packages(),
      install_requires=['nltk', 'pymystem3', 'pymorphy2', 'tensorflow-gpu', 'numpy', 'keras-bert', 'keras', 'yandex.translate', 'gensim', 'annoy', 'six'],
      zip_safe=False)