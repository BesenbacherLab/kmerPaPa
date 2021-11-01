"""
Entry-point module, in case you use `python -m kmerpapa`.

"""
import sys

from kmerpapa.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
