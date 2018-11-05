from setuptools import setup, Extension, Command, find_packages
import os,sys,glob

def read(fname):
    """Quickly read in the README.md file."""
    return open(os.path.join(os.path.dirname(__file__),fname)).read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

dist = setup(name='ximm_emulator',
             install_requires=['numpy','scipy','george'],
             version='1.0',
             packages=find_packages(),
             include_package_data=True,
             description='Emulator for the matter correlation function build from the Aemulus simulations.',
             long_description=read('README.md'),
             author='Tom McClintock',
             author_email='mcclintock@bnl.gov',
             url='https://github.com/tmcclintock/Matter-Correlation-Function',
             cmdclass={'clean': CleanCommand})

