import os
from osgeo import gdal
import click

def split(input, output):
    '''
    :param input: imagery/climate_test.tif -- input image
    :param output: imagery/climate_split/climate_test -- ouput folder with prefix of file
    :return:
    '''
    tile_size_x = 256
    tile_size_y = 256

    ds = gdal.Open(input)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            com_string = "gdal_translate -srcwin " + str(i) + ", " + str(j) + ", " + str(
                tile_size_x) + ", " + str(tile_size_y) + " " + str(input) + " " + str(
                output) + str(i) + "_" + str(j) + ".tif"
            os.system(com_string)

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())

def init(input, output):
    split(input, output)

if __name__ == "__main__":
    init()
