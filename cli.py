import argparse

def fetch_args():
    """
    Fetches the arguments mentioned during CLI usage
    """
    parser = argparse.ArgumentParser(
        description='Runs one of the pipeline scripts, for a given input file.')
    parser.add_argument('-p', '--pipeline', default='ga',
                        help='Name of the region proposer pipeline: ga/pso (Default: ga)')
    parser.add_argument('-f', '--filename', default='road.jpg',
                        help='Name of the input image file (Default: road.jpg)')
    parser.add_argument('-m', '--mode', default='unit',
                        help='Type of running mode: unit/batch (Default: unit)')
    parser.add_argument('-l', '--log', action="count", default=False, help='Logs output if specified')
    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    args = fetch_args()
    print(args.pipeline)
    print(args.filename)
    print(args.mode)
