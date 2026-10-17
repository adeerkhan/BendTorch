"""demo/mnist.bend against its PyTorch twin (reference/mnist_ref.py) for 50 batches under each optimizer.

Pass: the mean training loss within the shared tolerance (oracle.py) and the same number of test hits. Longer runs
drift apart slowly (the two sum the products in different orders, and Adam magnifies that); NOTES.md has the numbers.

    reference/.venv/bin/python reference/test_mnist.py .scratch/mnist
"""

import re
import subprocess
import sys

import torch

from oracle import close

RUNS = [("sgd", "0.1"), ("momentum", "0.05"), ("adam", "0.001"), ("adamw", "0.001")]
LINE = re.compile(r"epoch 1 train_loss=(\S+) test_correct=(\d+)/10000")


def parse(out: str) -> tuple[float, int]:
    m = LINE.search(out)
    if not m:
        raise SystemExit(f"no epoch line in:\n{out}")
    return float(m.group(1)), int(m.group(2))


def main():
    binary = sys.argv[1]
    failures: list[str] = []
    for opt, lr in RUNS:
        bend = parse(subprocess.run([binary, "1", "50", opt, lr], check=True, capture_output=True, text=True).stdout)
        ref = parse(subprocess.run([sys.executable, "reference/mnist_ref.py", "1", "50", opt, lr], check=True,
                                   capture_output=True, text=True).stdout)
        close(f"{opt} loss", torch.tensor([bend[0]]).numpy(), torch.tensor([ref[0]]), failures)
        if bend[1] != ref[1]:
            failures.append(f"{opt}: {bend[1]} hits, torch {ref[1]}")
        print(f"mnist {opt}: loss {bend[0]} (torch {ref[0]}), hits {bend[1]} (torch {ref[1]})")
    # The old CLI defaults to 100; an explicit run-time 100 must print identical results (apart from timing).
    old = subprocess.run([binary, "1", "3", "adam", "0.001"], check=True, capture_output=True, text=True).stdout
    explicit = subprocess.run([binary, "1", "3", "adam", "0.001", "100"], check=True, capture_output=True, text=True).stdout
    strip_time = lambda out: re.sub(r" train_seconds=[0-9.]+", "", out)
    if strip_time(old) != strip_time(explicit):
        failures.append("run-time 100 differs from the default batch size")
    for size, nb, opt, lr in [(37, 3, opt, lr) for opt, lr in RUNS] + [(8191, 0, "sgd", "0.01")]:
        args = ["1", str(nb), opt, lr, str(size)]
        bend = parse(subprocess.run([binary, *args], check=True, capture_output=True, text=True).stdout)
        ref = parse(subprocess.run([sys.executable, "reference/mnist_ref.py", *args], check=True,
                                   capture_output=True, text=True).stdout)
        close(f"batch {size} {opt} loss", torch.tensor([bend[0]]).numpy(), torch.tensor([ref[0]]), failures)
        if bend[1] != ref[1]:
            failures.append(f"batch {size} {opt}: {bend[1]} hits, torch {ref[1]}")
        print(f"mnist batch {size} {opt}: loss {bend[0]} (torch {ref[0]}), hits {bend[1]} (torch {ref[1]})")
    for args in [("1", "1", "sgd", "0.1", "0"), ("1", "1", "sgd", "0.1", "60001"),
                 ("1", "601", "sgd", "0.1"), ("1", "1", "sgd", "0.1", "bad")]:
        if subprocess.run([binary, *args], capture_output=True).returncode == 0:
            failures.append(f"invalid arguments accepted: {args}")
    if failures:
        print("\n".join(failures))
        sys.exit(1)
    print("mnist: PASS")


if __name__ == "__main__":
    main()
