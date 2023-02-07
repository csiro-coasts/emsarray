#!/usr/bin/env python3
"""
Tag the current commit for release
"""

import argparse
import configparser
import pathlib
import shlex
import subprocess
import sys
from typing import List, Optional

PROJECT = pathlib.Path(__file__).parent.parent


def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    add_options(parser)

    opts = parser.parse_args(args)
    opts.version = get_current_version()

    try:
        git_fetch(opts)
        tag_name = tag_version(opts)
        maybe_push(opts, tag_name)

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
    parser.add_argument(
        '--ref', type=str, default='origin/main',
        help=("The ref to tag. Defaults to 'origin/main'"))


def git_fetch(opts: argparse.Namespace) -> None:
    call('git', 'fetch', opts.remote)


def tag_version(opts: argparse.Namespace) -> str:
    call('git', 'show', opts.ref)
    if not yn("Is this the correct commit to tag?"):
        raise CommandError("Aborted")
    tag_name = f'v{opts.version}'
    call(
        'git', 'tag', tag_name, opts.ref,
        '-a', '-m', f'Version {opts.version}')
    return tag_name


def maybe_push(opts: argparse.Namespace, tag_name: str) -> None:
    call('git', 'show', tag_name)
    if not yn(f"Push tag {tag_name!r} to {opts.remote}?"):
        raise CommandError("Aborted")

    call('git', 'push', opts.remote, tag_name)


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


class CommandError(Exception):
    pass


if __name__ == '__main__':
    main()
