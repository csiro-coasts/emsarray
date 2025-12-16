#!/usr/bin/env python3

from livereload import Server, shell


def main() -> None:
    command = shell('make dirhtml')
    server = Server()
    server.watch('**/*.rst', command, delay=0.1)
    server.watch('**/*.py', command, delay=0.1)
    server.watch('**/*.ipynb', command, delay=0.1)
    server.watch('../src', command, delay=0.1)
    server.watch('../examples', command, delay=0.1)
    server.serve(root='./_build/dirhtml')


if __name__ == '__main__':
    main()
