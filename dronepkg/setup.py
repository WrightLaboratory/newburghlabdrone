from distutils.core import setup

__version__ = '1.0'

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_args = {
    'name': 'dronepkg',
    'author': 'Willy Tyndall',
    'license': 'BSD',
    'package_dir': {'dronepkg': 'dronepkg'},
    'packages': ['dronepkg'],
    'version': __version__,
    'setup_requires': ['pytest-runner', 'numpy'],
    'install_requires': requirements,
    'tests_require': requirements,

}

if __name__ == '__main__':
    setup(**setup_args)