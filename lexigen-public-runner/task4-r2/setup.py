from setuptools import Extension, setup

setup(
    name="lexigen-chacha",
    version="0.2.0",
    ext_modules=[
        Extension(
            "lexigen_chacha",
            ["lexigen_chacha.c"],
            libraries=["crypto"],
            extra_compile_args=["-O3", "-pthread"],
            extra_link_args=["-pthread"],
        )
    ],
)
