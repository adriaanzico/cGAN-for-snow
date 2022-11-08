import click
import rasterio
import numpy as np

# input = 'imagery/L8/2014/spring/L8_spring_2014_merged.tif'

def normalise (input, output):
    inp = rasterio.open(input)
    inp1 = np.nan_to_num(inp.read())
    normalised = ((inp1 - np.min(inp1)) / (np.max(inp1) - np.min(inp1)))
    try:
        # Create empty TIF image with dimensions of FIN but with name of FOUT.
        with rasterio.open(
                output,
                'w',
                driver='GTiff',
                height=inp.shape[1],
                width=inp.shape[2],
                count=inp.shape[0],
                dtype='float32',
                crs=inp.crs,
                transform=inp.transform,
        ) as dst:
            dst.write(normalised)
            print(f"File created: {output}")
    except IOError as e:
        print(f"Couldn't write a file at {output}. Error: {e}")

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())

def init(input, output):
    '''
    :param input: input filepath
    :param output: output filepath
    :return:
    '''
    normalise(input, output)

if __name__ == "__main__":
    init()
