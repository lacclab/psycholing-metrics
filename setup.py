from setuptools import setup

setup(
    name="text_metrics",
    version="1.1.12",
    packages=[
        "text_metrics",
        "text_metrics.surprisal_extractors",
        # pimentel_word_prob is a submodule of text_metrics
        "text_metrics.pimentel_word_prob",
        "text_metrics.pimentel_word_prob.wordsprobability",
        "text_metrics.pimentel_word_prob.wordsprobability.models",
        "text_metrics.pimentel_word_prob.wordsprobability.utils",
    ],
    url="https://github.com/lacclab/text-metrics",
    license="",
    author="Omer Shubi",
    author_email="omer.shubi@gmail.com",
    description="Utils for Eye Tracking Measurements",
    include_package_data=True,
    package_data={"": ["data/*.tsv"]},
    extras_require={"lm_zoo": ["lm-zoo"]},
    install_requires=[
        "pandas>2.1.0",
        "transformers>=4.40.1",
        "wordfreq>=3.0.3",
        "numpy>=1.20.3",
        # "torch",
        "spacy",
        "accelerate",
        "sentence-splitter",
    ],
)
