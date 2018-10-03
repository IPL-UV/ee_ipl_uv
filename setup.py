from setuptools import setup,find_packages
import ee_ipl_uv

setup(name='ee_ipl_uv',
      version=ee_ipl_uv.__version__,
      description='Functions to operate with GEE',
      author='Gonzalo Mateo Garcia',
      author_email='gonzalo.mateo-garcia@uv.es',
      packages=find_packages(exclude=["tests"]),
      install_requires=["numpy","requests",
                        "pandas", "earthengine-api"],
      include_package_data=True,
      zip_safe=False)
