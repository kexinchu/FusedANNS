import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        env = os.environ.copy()
        extra_cxx_flags = []

        # Help CMake find libomp on macOS when using Homebrew.
        if platform.system() == "Darwin":
            try:
                sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()
                env["SDKROOT"] = sdk_path
                cmake_args.append(f"-DCMAKE_OSX_SYSROOT={sdk_path}")
                stdlib_include = os.path.join(sdk_path, "usr/include/c++/v1")
                extra_cxx_flags.append(f"-isystem {stdlib_include}")
            except Exception:
                sdk_path = None
            libomp_prefix = os.environ.get("LIBOMP_PREFIX")
            if not libomp_prefix:
                conda_prefix = os.environ.get("CONDA_PREFIX")
                if conda_prefix:
                    conda_libomp = os.path.join(conda_prefix, "lib", "libomp.dylib")
                    conda_omp_header = os.path.join(conda_prefix, "include", "omp.h")
                    if os.path.exists(conda_libomp) and os.path.exists(conda_omp_header):
                        libomp_prefix = conda_prefix
            if not libomp_prefix:
                for candidate in ("/usr/local/opt/libomp", "/opt/homebrew/opt/libomp"):
                    if os.path.exists(candidate):
                        libomp_prefix = candidate
                        break
            if libomp_prefix:
                omp_include = os.path.join(libomp_prefix, "include")
                omp_lib = os.path.join(libomp_prefix, "lib")
                cmake_args += [
                    f"-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp -I{omp_include}",
                    f"-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp -I{omp_include}",
                    "-DOpenMP_C_LIB_NAMES=omp",
                    "-DOpenMP_CXX_LIB_NAMES=omp",
                    f"-DOpenMP_omp_LIBRARY={os.path.join(omp_lib, 'libomp.dylib')}",
                    f"-DCMAKE_EXE_LINKER_FLAGS=-L{omp_lib} -lomp",
                    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                ]
                env_cppflags = env.get('CPPFLAGS', '')
                env_ldflags = env.get('LDFLAGS', '')
                env['CPPFLAGS'] = f"{env_cppflags} -I{omp_include}".strip()
                env['LDFLAGS'] = f"{env_ldflags} -L{omp_lib}".strip()

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env['CXXFLAGS'] = '{} {} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            " ".join(extra_cxx_flags),
            self.distribution.get_version()
        ).strip()
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='pymswl',
    version='0.1',
    author='xxx',
    author_email='xxx',
    description='Python library for WeightLearning algorithm',
    long_description='',
    ext_modules=[CMakeExtension('pymswl')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
