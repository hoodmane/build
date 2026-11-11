# SPDX-License-Identifier: MIT

"""
Support for cross builds: building a wheel for a *host* platform that
differs from the *build* platform the frontend is running on.

The host platform is described by a :pep:`739` ``build-details.json`` document.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import shutil
import sys
import tempfile
import typing

from packaging.markers import Environment, host_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import host_platform_tags

from . import _ctx
from ._ctx import run_subprocess
from ._exceptions import BuildException


TYPE_CHECKING = False

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

    from ._types import JSONValue, StrPath
    from .env import Installer

    BuildDetails = Mapping[str, JSONValue]


# Environment variables conventionally used to describe a cross toolchain.
TOOLCHAIN_VARIABLES = (
    'CC',
    'CXX',
    'AR',
    'AS',
    'LD',
    'NM',
    'RANLIB',
    'STRIP',
    'OBJCOPY',
    'OBJDUMP',
    'READELF',
    'PKG_CONFIG',
    'CFLAGS',
    'CXXFLAGS',
    'LDFLAGS',
)


@dataclasses.dataclass(frozen=True)
class CrossBuildConfig:
    """Everything the frontend needs to know to perform a cross build."""

    build_details: BuildDetails
    """The parsed :pep:`739` ``build-details.json`` document of the host."""

    toolchain: Mapping[str, str]
    """The cross toolchain (``CC``, ``CXX``, ``AR``, ...)."""

    glibc_version: tuple[int, int] | None = None
    """glibc version of the host, if known. Not recorded in ``build-details.json``."""

    @property
    def platform_tags(self) -> list[str]:
        return list(host_platform_tags(self.build_details, glibc_version=self.glibc_version))

    @property
    def python_version(self) -> str:
        language = self.build_details['language']
        assert isinstance(language, dict)
        return str(language['version'])

    @property
    def implementation(self) -> str:
        implementation = self.build_details['implementation']
        assert isinstance(implementation, dict)
        return str(implementation['name'])

    @property
    def abis(self) -> list[str]:
        """ABI tags for the host interpreter, most specific first."""
        from packaging.tags import host_tags

        abis: list[str] = []
        for tag in host_tags(self.build_details, glibc_version=self.glibc_version):
            if tag.abi not in abis:
                abis.append(tag.abi)
        return abis

    @functools.cached_property
    def marker_environment(self) -> Environment:
        return host_environment(self.build_details)


def load_build_details(path: StrPath) -> BuildDetails:
    """Load and minimally validate a ``build-details.json`` document."""
    try:
        with open(path, 'rb') as f:
            details: JSONValue = json.load(f)
    except FileNotFoundError:
        msg = f'build-details.json not found: {path}'
        raise BuildException(msg) from None
    except json.JSONDecodeError as e:
        msg = f'Failed to parse {path}: {e}'
        raise BuildException(msg) from None
    if not isinstance(details, dict):
        msg = f'{path} does not contain a JSON object'
        raise BuildException(msg)
    schema_version = str(details.get('schema_version', ''))
    if not schema_version.startswith('1.'):
        msg = f'{path}: unsupported build-details.json schema_version {schema_version!r}'
        raise BuildException(msg)
    for key in ('language', 'implementation', 'abi'):
        if key not in details:
            msg = f'{path}: missing required key {key!r}'
            raise BuildException(msg)
    # PEP 739: a relative ``base_prefix`` is relative to the directory containing
    # the file, and all other paths are relative to ``base_prefix``. Resolve it so
    # the document stays valid when passed around or re-serialised elsewhere.
    base_prefix = details.get('base_prefix')
    if isinstance(base_prefix, str) and not os.path.isabs(base_prefix):
        details['base_prefix'] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), base_prefix))
    return details


def parse_glibc_version(value: str) -> tuple[int, int]:
    try:
        major, minor = value.split('.')
        return (int(major), int(minor))
    except ValueError:
        msg = f'Invalid glibc version {value!r}, expected MAJOR.MINOR (e.g. 2.28)'
        raise BuildException(msg) from None


def toolchain_from_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """
    Collect the cross toolchain description from the conventional environment
    variables (``CC``, ``CXX``, ``AR``, ...), applying ``overrides`` on top.
    """
    toolchain = {name: os.environ[name] for name in TOOLCHAIN_VARIABLES if name in os.environ}
    if overrides:
        toolchain.update(overrides)
    return toolchain


def parse_toolchain_args(args: Iterable[str] | None) -> dict[str, str]:
    """Parse ``NAME=VALUE`` command line arguments."""
    toolchain: dict[str, str] = {}
    for arg in args or ():
        name, sep, value = arg.partition('=')
        if not sep or not name:
            msg = f'Invalid toolchain setting {arg!r}, expected NAME=VALUE'
            raise BuildException(msg)
        toolchain[name] = value
    return toolchain


def filter_host_requirements(requirements: Collection[str], cross: CrossBuildConfig) -> list[str]:
    """
    Evaluate the markers of ``requirements`` against the host environment and
    return the requirements that apply, with their markers stripped (so that
    the installer does not re-evaluate them against the build environment).
    """
    result: list[str] = []
    for requirement_string in sorted(requirements):
        try:
            requirement = Requirement(requirement_string)
        except InvalidRequirement as e:
            msg = f'Invalid host requirement {requirement_string!r}: {e}'
            raise BuildException(msg) from None
        if requirement.marker is not None:
            environment = typing.cast('Mapping[str, str]', cross.marker_environment)
            if not requirement.marker.evaluate(environment, context='requirement'):
                continue
            requirement.marker = None
        result.append(str(requirement))
    return result


def _uv_python_platform(cross: CrossBuildConfig) -> str:
    """Map the host platform to a ``uv pip install --python-platform`` value."""
    from packaging import _build_details

    host = _build_details.host_platform(cross.build_details)
    machine = host.machine
    match host.system:
        case 'Linux':
            if host.libc == 'musl':
                return f'{machine}-unknown-linux-musl'
            if cross.glibc_version is not None:
                return f'{machine}-manylinux_{cross.glibc_version[0]}_{cross.glibc_version[1]}'
            return f'{machine}-unknown-linux-gnu'
        case 'Darwin':
            return f'{"aarch64" if machine == "arm64" else machine}-apple-darwin'
        case 'Windows':
            return f'{machine.lower().replace("amd64", "x86_64").replace("arm64", "aarch64")}-pc-windows-msvc'
        case _:
            msg = f'uv does not support installing packages for host platform {host.sysconfig_platform!r}'
            raise BuildException(msg)


def install_host_requirements(
    requirements: Collection[str],
    target_directory: str,
    cross: CrossBuildConfig,
    *,
    installer: Installer = 'pip',
    constraints_txt_path: StrPath | None = None,
) -> None:
    """
    Install binary wheels of ``requirements`` for the host platform into
    ``target_directory``, which is a plain directory (not a virtual environment).
    """
    if not requirements:
        return

    _ctx.log(
        'Installing host packages for cross build:\n' + '\n'.join(f'- {r}' for r in sorted(requirements)),
        kind=('step',),
    )

    if installer == 'uv':
        _install_host_requirements_uv(requirements, target_directory, cross, constraints_txt_path)
    else:
        _install_host_requirements_pip(requirements, target_directory, cross, constraints_txt_path)


def _install_host_requirements_pip(
    requirements: Collection[str],
    target_directory: str,
    cross: CrossBuildConfig,
    constraints_txt_path: StrPath | None,
) -> None:
    cmd = [sys.executable, '-m', 'pip']
    if (verbosity := _ctx.verbosity) > 1:
        cmd += [f'-{"v" * (verbosity - 1)}']
    cmd += [
        'install',
        '--target',
        target_directory,
        '--only-binary=:all:',
        '--no-compile',
        '--no-input',
        '--python-version',
        cross.python_version,
        '--implementation',
        _pip_implementation(cross.implementation),
    ]
    for platform_tag in cross.platform_tags:
        cmd += ['--platform', platform_tag]
    for abi in cross.abis:
        cmd += ['--abi', abi]
    with tempfile.NamedTemporaryFile(
        'w', prefix='build-host-requirements-', suffix='.txt', delete=False, encoding='utf-8'
    ) as requirement_file:
        requirement_file.write('\n'.join(requirements))
    try:
        cmd += ['-r', os.path.abspath(requirement_file.name)]
        if constraints_txt_path:
            cmd += ['-c', str(constraints_txt_path)]
        run_subprocess(cmd)
    finally:
        os.unlink(requirement_file.name)


def _pip_implementation(name: str) -> str:
    return {'cpython': 'cp', 'pypy': 'pp', 'ironpython': 'ip', 'jython': 'jy'}.get(name, name)


def _install_host_requirements_uv(
    requirements: Collection[str],
    target_directory: str,
    cross: CrossBuildConfig,
    constraints_txt_path: StrPath | None,
) -> None:
    try:
        import uv

        uv_bin = uv.find_uv_bin()
    except (ModuleNotFoundError, FileNotFoundError):
        uv_bin = shutil.which(os.environ.get('UV') or 'uv')
        if uv_bin is None:
            msg = 'uv executable not found'
            raise BuildException(msg) from None

    cmd = [uv_bin, 'pip']
    if (verbosity := _ctx.verbosity) > 1:
        cmd += [f'-{"v" * min(2, verbosity - 1)}']
    cmd += [
        'install',
        '--target',
        target_directory,
        '--only-binary',
        ':all:',
        '--python-version',
        cross.python_version,
        '--python-platform',
        _uv_python_platform(cross),
        '--python',
        sys.executable,
        *requirements,
    ]
    if constraints_txt_path:
        cmd += ['-c', str(constraints_txt_path)]
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'VIRTUAL_ENV')}
    run_subprocess(cmd, env=env)
