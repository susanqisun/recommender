from setuptools import setup

import os
...
port = int(os.environ.get('PORT', 5000))
...
app.run(host='0.0.0.0', port=port, debug=True)

setup(
    name='flasky',
    packages=['flaskr'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)
