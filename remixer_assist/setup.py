# To be finished
import setuptools

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='remixer_assist',
    version='0.0.1',
    author='José Edivaldo',
    author_email='jose.edivaldo.fisica@gmail.com',
    description='An app that assist in creating remixes of group of songs'
)