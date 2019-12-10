import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=0.0)
    args = parser.parse_args()

    args.tau