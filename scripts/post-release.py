#!/usr/bin/env python
"""
Get the project ready for development after a release
"""
import argparse
import configparser
import pathlib
import re
import shlex
import subprocess
import sys
from typing import List, Optional

PROJECT = pathlib.Path(__file__).parent.parent


release_notes_index = pathlib.Path('docs/releases/index.rst')
release_notes_dev = pathlib.Path('docs/releases/development.rst')
release_notes_content = """
=============================
Next release (in development)
=============================

* ...
""".lstrip()


def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    add_options(parser)

    opts = parser.parse_args(args)
    opts.version = get_current_version()

    print("Preparing for development")

    try:
        git_check(opts)
        branch_name = git_branch(opts)
        development_release_notes()
        git_commit(opts)
        maybe_push(opts, branch_name)

    except CommandError as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)

    except KeyboardInterrupt:
        sys.exit(1)


def add_options(parser: argparse.ArgumentParser) -> None:
    parser.description = __doc__
    parser.set_defaults(version='placeholder')
    parser.add_argument(
        '--remote', type=str, default='origin',
        help=("The name of the csiro-coasts/emsarray remote. Defaults to 'origin'"))


def git_check(opts: argparse.Namespace) -> None:
    status = output('git', 'status', '--porcelain', '--untracked-files=no')
    if status != b'':
        raise CommandError(f'Working tree is dirty, aborting!\n{status.decode()}')
    call('git', 'fetch', opts.remote)


def git_branch(opts: argparse.Namespace) -> str:
    branch_name = f'post-{opts.version}-release'
    call('git', 'checkout', '-b', branch_name, f'{opts.remote}/main')
    return branch_name


def development_release_notes() -> None:
    release_notes_dev.write_text(release_notes_content)
    toctree_re = re.compile(r'^\.\. toctree::$\n(^ +.*$\n)*^$\n', re.MULTILINE)
    replace_in_file(
        release_notes_index,
        toctree_re,
        r'\g<0>   ' + release_notes_dev.stem + '\n')
    call('git', 'add', str(release_notes_dev), str(release_notes_index))


def git_commit(opts: argparse.Namespace) -> None:
    call('git', '--no-pager', 'diff', '--cached')
    call(
        'git', 'commit',
        '-m', f'Prepare for development after version {opts.version} release')


def maybe_push(
    opts: argparse.Namespace,
    ref_name: str,
) -> None:
    if not yn(f"Push {ref_name!r} to {opts.remote}?"):
        raise CommandError("Aborted")

    call('git', 'push', opts.remote, ref_name)


def get_current_version() -> str:
    parser = configparser.ConfigParser()
    with open(PROJECT / 'setup.cfg') as f:
        parser.read_file(f)
    return parser.get('metadata', 'version')


def call(*args: str) -> None:
    print('$', shlex.join(args))
    try:
        subprocess.check_call(args, cwd=PROJECT)
    except subprocess.CalledProcessError as err:
        raise CommandError("Aborted") from err


def output(*args: str) -> bytes:
    return subprocess.check_output(args, cwd=PROJECT)


def yn(
    prompt: str,
    default: Optional[bool] = None,
) -> bool:
    examples = {True: '[Yn]', False: '[yN]', None: '[yn]'}[default]
    prompt = f'{prompt.strip()} {examples} '

    while True:
        response = input(prompt).lower()
        if response in {'y', 'yes'}:
            return True
        elif response in {'n', 'no'}:
            return False
        elif response == '' and default is not None:
            return default
        else:
            print("Invalid response '{response!r}'. Please enter y or n.")


def replace_in_file(
    path: pathlib.Path,
    pattern: re.Pattern,
    replacement: str,
) -> None:
    full_path = PROJECT / path
    old_content = full_path.read_text()
    if pattern.search(old_content) is None:
        raise CommandError(f"Error updating {path}: pattern not found!")
    new_content = pattern.sub(replacement, old_content)
    full_path.write_text(new_content)


class CommandError(Exception):
    pass


if __name__ == '__main__':
    main()
