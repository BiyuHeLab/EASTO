from setuptools import setup, find_packages 

#List of requirements\
requirements = ['numpy >=1.21.2', 'scipy >=1.7.3', 'pandas>=1.3.5']

setup(
	name = "EASTO",
	version = "1.0.0",
	description = "Functions for processing EASTO exp data and images",
	packages = find_packages(), 
	install_requires = requirements)