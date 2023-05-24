import argparse


if __name__ == '__main__':
    raise NotImplementedError()
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="maxcut")
    parser.add_argument("-n", type=int, default=16)
    parser.add_argument("-p", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-b", "--backend", type=str, default="sv")
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--no-aer", dest="aer", default=True, action="store_false")
    parser.add_argument("--noise", type=str, default="ideal")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--ansatz", type=str, default="qaoa")
    parser.add_argument("--p1", type=float, default=0.001)
    parser.add_argument("--p2", type=float, default=0.005)
    parser.add_argument("--beta-steps", type=int, default=50)
    parser.add_argument("--gamma-steps", type=int, default=100)
    parser.add_argument("--mitiq", type=str, default=None)
    parser.add_argument("--mitiq-config", type=int, default=0)

    args = parser.parse_args()
