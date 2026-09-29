import argparse

from architecture import CPU

parser = argparse.ArgumentParser()

parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("ROM")


if __name__ == "__main__":
    args = parser.parse_args()
    cpu = None
    if args.verbose:
        cpu = CPU(args.ROM, True)
    else:
        cpu = CPU(args.ROM)
    cpu.main_loop()
