import rasterio
from rasterio.enums import Resampling
import click

# input = 'imagery/climate_data.tif'
# reference = 'imagery/landsat8.tif'
# output = 'imagery/climate_test_resampled.tif'

def resample (input, reference, output):
    image1 = rasterio.open(input)
    image_ref = rasterio.open(reference)

    image1_arr = image1.read(out_shape=(image1.count, image_ref.shape[0], image_ref.shape[1]), resampling=Resampling.bilinear)

    try:
        # Create empty TIF image with dimensions of FIN but with name of FOUT.
        with rasterio.open(
                output,
                'w',
                driver='GTiff',
                height=image1_arr.shape[1],
                width=image1_arr.shape[2],
                count=image1_arr.shape[0],
                dtype='float32',
                crs=image_ref.crs,
                transform=image_ref.transform,
        ) as dst:
            dst.write(image1_arr)
            print(f"File created: {output}")
    except IOError as e:
        print(f"Couldn't write a file at {output}. Error: {e}")

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('reference', type=click.Path(exists=True))
@click.argument('output', type=click.Path())

def init(input, reference, output):
    resample(input, reference, output)

if __name__ == "__main__":
    init()

