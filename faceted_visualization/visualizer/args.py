import ast
from argparse import ArgumentParser
import os


def parse_args(config = None):

    if config is None:
        cwd = os.path.dirname(__file__)

        with open(os.path.join(cwd, "config", "cli_args.json")) as f:
            config = ast.literal_eval(f.read())
            # print(config)

    parser = ArgumentParser()
    for key, value in config.items():
        parser.add_argument(
                            value['long_flag'],
                            action=value.get('action', 'store'),
                            required=value.get('is_required', False),
                            help=value['help_message'],
                            default=value.get('default', None))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parse_args()


