from setuptools import setup

__author__ = "Puck van Gerwen"
__copyright__ = "Copyright 2022"
__credits__ = ["Puck van Gerwen et al, Mach. Learn. Sci. Tech.:2022(3), 045005"]
__license__ = "MIT"
__version__ = "0"
__maintainer__ = "Puck van Gerwen"
__email__ = "puck.vangerwen@epfl.ch"
__description__ = "B2R2 reaction representations"
__url__ = "https://github.com/lcmd-epfl/b2r2-reaction-rep"

def requirements():
    with open('requirements.txt') as f:
        return [line.rstrip() for line in f]

# use README.md as long description
def readme():
    with open('README.md') as f:
        return f.read()

setup(

        name="reactionreps",
        packages=[
            'reactionreps',
            'data'
            ],

        # metadata
        version=__version__,
        author=__author__,
        author_email=__email__,
        url=__url__,
        platforms = 'Any',
        description = __description__,
        long_description = readme(),
        keywords = ['Machine Learning', 'Quantum Chemistry'],
        classifiers = [],
        setup_requires = ['numpy'],
        install_requires = requirements(),
)
