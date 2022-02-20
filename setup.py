from setuptools import setup, find_packages

setup(
    name='ml-tools',
    version='0.0.1',
    description='Wrapper tools for machine learning workflows',
    url='https://github.com/webbhalsa/kry-algos-notebooks.git',
    author='Simon Celinder',
    author_email='simon.mindfulprofessionals@gmail.com',
    license='None',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'examples']),
    install_requires=[
        "jupyter==1.0.0",
        "numpy==1.21",
        "pandas==1.4.1",
        "shap-hypetune==0.1.1",
        "ipython==7.31.0",
        "lightgbm==3.3.2",
        "catboost==1.0.4",
        "cufflinks==0.17.3",
        "seaborn==0.11.2",
        "matplotlib==3.5.1",
        "plotly==5.4.0",
        "chart-studio==1.1.0",
        "optuna==2.10.0"
    ],
    extras_require={
        'test': [
            'pytest==6.2.5'
        ],
    },
    tests_require=['nose'],
    zip_safe=False
)
