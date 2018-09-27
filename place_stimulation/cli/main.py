from expipecli.utils.plugin import IPlugin
from place_stimulation.imports import *
from . import analysis
from . import openephys as OE
from . import intan as I
from . import intan_ephys as IE
from . import electrical_stimulation


def reveal():
    """
    This imports all plugins when loading expipe-cli.
    """
    pass


class ElectroPlugin(IPlugin):
    def attach_to_cli(self, cli):
        @cli.group(short_help='Tools related to Open Ephys.')
        @click.help_option('-h', '--help')
        @click.pass_context
        def openephys(ctx):
            pass

        @cli.group(short_help='Tools related to Intan.')
        @click.help_option('-h', '--help')
        @click.pass_context
        def intan(ctx):
            pass

        @cli.group(short_help='Tools related to the combination of Open Ephys and Intan.')
        @click.help_option('-h', '--help')
        @click.pass_context
        def intan_ephys(ctx):
            pass

        analysis.attach_to_cli(cli)
        OE.attach_to_cli(openephys)
        I.attach_to_cli(intan)
        IE.attach_to_cli(intan_ephys)
        electrical_stimulation.attach_to_cli(cli)
