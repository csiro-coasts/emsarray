#!/usr/bin/env python3

from livereload import Server, shell


def main() -> None:
    server = Server()
    server.watch('**/*.rst', shell('make html'), delay=0.1)
    server.watch('../src', shell('make html'), delay=0.1)
    server.serve(root='./_build/html')


if __name__ == '__main__':
    main()
