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
tests_dependencies = ["networkx", "scikit-learn"],

version = get_version(package_name+"/__init__.py")
repo_url = os.path.join(author_url, package_name)

setup(name=package_name,
      version=version,
      description='Benchmarking tools for Python',
      long_description=fix_imgabspath(repo_url, read('README.rst')),
      url=repo_url,
      author='Divakar Roy',
      author_email='droygatech@gmail.com',
      platforms=['any'],
      license='MIT',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: System :: Benchmark',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Operating System :: OS Independent',
          ],
      keywords='benchmarking performance timing timeit',
      install_requires=["pandas", "numpy", "py-cpuinfo", "tqdm", "psutil", "matplotlib"],
      extras_require={
          'optional': optional_dependencies,
          'development': optional_dependencies + development_dependencies,
          'maintainer': optional_dependencies + development_dependencies + maintainer_dependencies,
          'test': tests_dependencies
      },
      include_package_data=True,
      zip_safe=False)
