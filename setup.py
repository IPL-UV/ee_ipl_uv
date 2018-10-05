from setuptools import setup,find_packages

setup(name='ee_ipl_uv',
      version='0.1',
      description='Functions to operate with GEE',
      author='Gonzalo Mateo Garcia',
      author_email='gonzalo.mateo-garcia@uv.es',
      packages=find_packages(exclude=["tests"]),
      install_requires=["numpy","requests",
                        "pandas", "earthengine-api"],
      include_package_data=True,
      package_data={
            # If any package contains *.txt or *.rst files, include them:
            '': ['*.json']
      },
      zip_safe=False)
