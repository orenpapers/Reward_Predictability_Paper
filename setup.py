import setuptools

setuptools.setup(
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        # "bert-extractive-summarizer",
        "matplotlib", "mne",
        "scikit-learn==0.24.2", "seaborn",
        "huggingface_hub",
        "numpy",
        'palettable',
        # "numpy==1.20.0",
        "scipy", #"word2vec
        "pandas", #'gensim',
        "simcse", "binsreg",
        'yellowbrick', "joblib",
        #"joblib==0.11",
        "tensorflow",
        "bert-extractive-summarizer",
        "nltk", "gensim", "keras",
        "nilearn","nistats","nibabel", 'happytransformer',#==2.1.0',
        'spacy','sentence-transformers','huggingface',
        'simpletransformers', "transformers",
        # "transformers==3.1", # on mac need to run curl https://sh.rustup.rs -sSf | sh
        # "transformers==4.6.1", # on mac need to run curl https://sh.rustup.rs -sSf | sh
        'wordfreq','tikreg',"nilearn",
        # 'mxnet-mkl',
        'torch','numpy',
        'flair','textblob', 'openai',
        'datasets',
        # "wandb",
        # "nistats"
        # "brainiak"
        # 'llvmlite',
        # 'allennlp',
        # 'top2vec',
        # 'bertopic'

    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        'License :: Other/Proprietary License',
        'Private :: Do Not Upload',
    ],
)