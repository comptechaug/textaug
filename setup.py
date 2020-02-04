from setuptools import setup, find_packages
from os.path import join, dirname

with open('requirements.txt', 'r') as file:
    requirements = [x.strip() for x in file.readlines()]

setup(name='TextAugRus',
      version='0.1.0',
      description='Little package for text augmentation',
      author_email='comptechaugteam@gmail.com',
      url='https://github.com/comptechaug/textaug',
      author='TextAugmentationTeam',
      license='MIT',
      packages=['TextAugRus'],
      include_package_data=True,
      install_requires=requirements,
      classifiers=[
          'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
          'Intended Audience :: Developers',      # Define that your audience are developers
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',   # Again, pick a license
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
        ],
      zip_safe=False)
