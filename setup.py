import re
import ast
import os
from setuptools import setup, find_packages


def get_version(init_file):
    _version_re = re.compile(r'__version__\s+=\s+(.*)')
    with open(init_file, 'rb') as f:
        version = str(ast.literal_eval(_version_re.search(f.read().decode('utf-8')).group(1)))
    return version


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def fix_imgabspath(repo_url, in_str, branch='master'):
    repo_url_for_images = repo_url.replace('https://github.com/', 'https://raw.githubusercontent.com/')
    return in_str.replace('./docs/source/', os.path.join(repo_url_for_images, branch, 'docs/source/'))


author_url = 'https://github.com/droyed'
package_name = "benchit"
optional_dependencies = []
development_dependencies = ["sphinx"]
maintainer_dependencies = ["twine"]

version = get_version(package_name+"/__init__.py")
repo_url = os.path.join(author_url, package_name)

setup(name=package_name,
      version=version,
      description='Benchmarking tools for Python',
      long_description=fix_imgabspath(repo_url, read('README.rst')),
      url=repo_url,
      author='Divakar Roy',
      author_email='droygatech@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=["pandas", "numpy", "py-cpuinfo", "tqdm", "psutil", "matplotlib"],
      extras_require={
          'optional': optional_dependencies,
          'development': optional_dependencies + development_dependencies,
          'maintainer': optional_dependencies + development_dependencies + maintainer_dependencies
      },
      zip_safe=False)
