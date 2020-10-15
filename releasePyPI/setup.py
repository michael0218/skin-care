
from setuptools import find_packages, setup

setup(
    name = 'skin-APIv',
    version = '1.1',
    description="The API for melanoma image skin level prediction.",
    packages= find_packages(),
    py_modules=['skinAPI'],  # pip安裝完之後,要import時的名稱
    author="Claire",
    license="LGPL"
)