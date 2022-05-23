import setuptools

try:
    import pip

    from packaging import version
    if version.parse(pip.__version__) < version.parse("21.3"):
        # Can be removed when pip editable user installs are fixed
        # https://github.com/pypa/pip/issues/7953
        import site
        import sys
        site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

except ImportError:
    pass

setuptools.setup()
