'''
Main class for CLI functionality.
'''
import argparse
from brain2brain.experiments import Experiments
from brain2brain.utils import Utils

def main():
    '''
    Main function for __main__.py. This is the entry point in the CLI.
    '''
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input_directory", help="Directory that contains\
    #                     GeoJSON files with the format YYYYMMDD.geojson.")
    # parser.add_argument("start_date", help="Start date (format YYYYMMDD).", type=int)
    # parser.add_argument("end_date", help="End date (inclusive, format YYYYMMDD).", type=int)
    # parser.add_argument("geohash_precision", help="Geohash precision level\
    #                     (between 1 and 12, default: 8)", nargs='?', default=8, type=int)
    # parser.add_argument("--output_path", "-o", help="Output file path for resulting\
    #                     JSON (optional). If output_path is\
    #                     not set, the results will only be printed, but not saved.")
    # args = parser.parse_args()

    # if args.output_path is not None:
    #     json_output = HeatMapper.create_heatmap(args.input_directory, args.start_date,
    #                                             args.end_date, args.output_path,
    #                                             args.geohash_precision)
    # else:
    #     json_output = HeatMapper.create_heatmap(args.input_directory, args.start_date,
    #                                             args.end_date, "", args.geohash_precision)

    # # Print JSON to CLI.
    # json_object = json.loads(json_output)
    # print(json.dumps(json_object, indent=4))

if __name__ == '__main__':
    main()