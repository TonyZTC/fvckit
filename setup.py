from distutils.core import setup

# patch distutils if it can't cope with the "classifiers" or
# "download_url" keywords
from sys import version
if version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None

setup(
    name='FVCKIT',
    version='1.2.4',
    author='Ewald Enzinger',
    author_email='ewald.enzinger@entn.at',
    packages=['fvckit', 'fvckit.bosaris', 'fvckit.frontend'],
    url='https://github.com/entn-at/fvckit',
    download_url='http://pypi.python.org/pypi/Fvckit/',
    license='LGPL',
    platforms=['Linux, Windows', 'MacOS'],
    description='Forensic Voice Comparison package.',
    long_description=open('README.txt').read(),
    install_requires=[
        "mock>=1.0.1",
        "nose>=1.3.4",
        "numpy>=1.10.4",
        "pyparsing >= 2.0.2",
        "python-dateutil >= 2.2",
        "scipy>=0.12.1",
        "six>=1.8.0",
        "matplotlib>=1.3.1",
	"PyYAML>=3.11",
	"h5py>=2.5.0",
	"pandas>=0.16.2"
    ],
    classifiers=['Development Status :: 4 - Beta',
                 'Environment :: Console',
                 'Environment :: MacOS X',
                 'Environment :: Win32 (MS Windows)',
                 'Environment :: X11 Applications',
                 'Intended Audience :: Education',
                 'Intended Audience :: Legal Industry',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
                 'Natural Language :: English',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft',
                 'Operating System :: Other OS',
                 'Operating System :: POSIX',
                 'Programming Language :: Python :: 3.5',
                 'Topic :: Multimedia :: Sound/Audio :: Speech',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence']
)

