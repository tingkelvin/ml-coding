import setuptools

setuptools.setup(

    install_requires=[
        'pandas',
        'scikit-learn',
        'xgboost',
        'cloudml-hypertune',
        'google-cloud-storage'
    ],

    packages=setuptools.find_packages())
