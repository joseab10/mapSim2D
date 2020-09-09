#! /usr/bin/env python

from map_simulator.map_simulator_2d import MapSimulator2D

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate a ROSbag file from a simulated robot trajectory.")

    parser.add_argument('-i', '--input', action='store', help='Input JSON robot config file', type=str, required=True)
    parser.add_argument('-o', '--output', action='store', help='Output ROSbag file', type=str, required=False)

    parser.add_argument('-p', '--preview', action='store_true')
    parser.add_argument('-s', '--search_paths', action='store', type=str,
                        default='.:robots:maps:src/map_simulator/robots:src/map_simulator/maps',
                        help='Search paths for the input and include files separated by colons (:)')

    args, override_args = parser.parse_known_args()

    override_str = None

    if len(override_args) > 0:
        override_str = '{'
        for arg in override_args:
            arg_keyval = arg.split(":=")
            override_str += '"' + str(arg_keyval[0]) + '":' + str(arg_keyval[1]) + ','

        override_str = override_str[0:-1] + "}"

    simulator = MapSimulator2D(args.input, args.search_paths, override_params=override_str)
    simulator.convert(args.output, display=args.preview)
