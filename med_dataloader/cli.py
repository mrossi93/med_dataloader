"""Console script for med_dataloader."""

import sys
import click
from med_dataloader import generate_dataset

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-d", "--data_dir",
              required=True,
              type=click.Path(exists=True,
                              file_okay=False,
                              dir_okay=True),
              help="The folder that contains data to load.")
@click.option("-A", "--A_label",
              required=True,
              type=click.STRING,
              help="Name of folder A inside 'data_dir'")
@click.option("-B", "--B_label",
              required=True,
              type=click.STRING,
              help="Name of folder B inside 'data_dir'")
@click.option("-s", "--size_input",
              required=True,
              type=click.INT,
              help="Input size of images (s x s)")
@click.option("--bounds_A",
              default=(None, None),
              show_default=True,
              type=(float, float),
              help="Boundaries for normalization of A images.")
@click.option("--bounds_B",
              default=(None, None),
              show_default=True,
              type=(float, float),
              help="Boundaries for normalization of B images.")
@click.option("-n", "--num_classes",
              default=None,
              show_default=True,
              required=False,
              type=click.INT,
              help="""Number of classes in case B folder contains categorical
              labels""")
@click.option("-o", "--output_dir",
              default=None,
              show_default=True,
              type=click.Path(exists=False,
                              file_okay=False,
                              dir_okay=True),
              help="""The folder that contains processed data. If set to 'None'
              (default) is equal to 'data_dir' with '_TF' suffix.""")
def main(data_dir,
         output_dir,
         a_label,
         b_label,
         size_input,
         bounds_a,
         bounds_b,
         num_classes):
    """Console script for med_dataloader."""
    click.echo(f"""\rGenerating dataset with following parameters:
    data_dir: {data_dir},
    A_label: {a_label},
    B_label: {b_label},
    Size: {size_input},
    Boundaries A: {bounds_a},
    Boundaries B: {bounds_b},
    Num classes: {num_classes}""")

    if bounds_a[0] is None:
        bounds_a = None
    if bounds_b[0] is None:
        bounds_b = None

    if num_classes is not None:
        is_B_categorical = True
    else:
        is_B_categorical = False

    generate_dataset(data_dir=data_dir,
                     imgA_label=a_label,
                     imgB_label=b_label,
                     input_size=size_input,
                     output_dir=output_dir,
                     is_B_categorical=is_B_categorical,
                     num_classes=num_classes,
                     norm_boundsA=bounds_a,
                     norm_boundsB=bounds_b,
                     )

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
