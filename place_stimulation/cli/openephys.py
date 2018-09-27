from place_stimulation.imports import *
from expipe_plugin_cinpla.tools import action as action_tools
from place_stimulation.tools import config
from datetime import timedelta



def validate_depth(ctx, param, depth):
    try:
        out = []
        for pos in depth:
            key, num, z, unit = pos.split(' ', 4)
            out.append((key, int(num), float(z), unit))
        return tuple(out)
    except ValueError:
        raise click.BadParameter('Depth need to be contained in "" and ' +
                                 'separated with white space i.e ' +
                                 '<"key num depth physical_unit"> (ommit <>).')


def attach_to_cli(cli):
    @cli.command('process',
                 short_help='Generate a klusta .dat and .prm files from openephys directory.')
    @click.argument('action-id', type=click.STRING)
    @click.option('--prb-path',
                  type=click.STRING,
                  help='Path to probefile, assumed to be in expipe config directory by default.',
                  )
    @click.option('--openephys-path',
                  type=click.STRING,
                  help='Path to openeophys dir, if none it is deduced from action id.',
                  )
    @click.option('--exdir-path',
                  type=click.STRING,
                  help='Path to desired exdir directory, if none it is deduced from action id.',
                  )
    @click.option('--klusta-filter',
                  is_flag=True,
                  help='Let klusta filter or not. Default = False',
                  )
    @click.option('--pre-filter',
                  is_flag=True,
                  help='Pre filter or not, replaces klusta filter. Default = True',
                  )
    @click.option('--filter-low',
                  type=click.INT,
                  default=300,
                  help='Low cut off frequencey. Default is 300 Hz',
                  )
    @click.option('--filter-high',
                  type=click.INT,
                  default=6000,
                  help='High cut off frequencey. Default is 6000 Hz',
                  )
    @click.option('--filter-order',
                  type=click.INT,
                  default=3,
                  help='Butterworth filter order N. Default is N=3. For acausal filters (filter type "filtfilt"), the effective filter order is N*2')
    @click.option('--filter-function',
                  type=click.Choice(['filtfilt', 'lfilter']),
                  default='filtfilt',
                  help='Filter function. The default "filtfilt" corresponds to a forward-backward filter operation, "lfilter" a forward filter. NOTE: does not affect filtering with klusta.')
    @click.option('--use-single-threshold',
                  type=click.Choice(['True', 'False']),
                  default='True',
                  help='Use the same threshold across channels. Default is True')
    @click.option('--threshold-strong-std-factor',
                  type=click.FLOAT,
                  default=4.5,
                  help='Strong spike detection threshold. Default is 4.5',
                  )
    @click.option('--threshold-weak-std-factor',
                  type=click.FLOAT,
                  default=2,
                  help='Weak spike detection threshold. Default is 2',
                  )
    @click.option('--common-ref',
                  type=click.Choice(['car', 'cmr', 'none']),
                  default='cmr',
                  help='Apply Common average/median (car/cmr) exreferencing. Default is "cmr"')
    @click.option('--split-probe',
                  type=click.INT,
                  default=16,
                  help='Splits referencing in 2 at selected channel (for CAR/CMR). Default is 16',
                  )
    @click.option('--ground', '-g',
                  multiple=True,
                  help='Ground selected channels')
    @click.option('--no-klusta',
                  is_flag=True,
                  help='Not run klusta on dataset.',
                  )
    @click.option('--no-spikes',
                  is_flag=True,
                  help='Disable convertion of spikes to exdir.',
                  )
    @click.option('--no-lfp',
                  is_flag=True,
                  help='Disable convertion of LFP to exdir.',
                  )
    @click.option('--no-tracking',
                  is_flag=True,
                  help='Disable registering of tracking data to exdir.',
                  )
    @click.option('--no-local',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    @click.option('--no-preprocess',
                  is_flag=True,
                  help='Preprocess data.',
                  )
    @click.option('--shutter-channel',
                  type=click.INT,
                  default=0,
                  help='TTL channel for shutter events to sync tracking',
                  )
    def process_openephys(action_id, prb_path, pre_filter,
                          klusta_filter, filter_low, filter_high,
                          filter_order, filter_function,
                          use_single_threshold,
                          threshold_strong_std_factor,
                          threshold_weak_std_factor,
                          common_ref, ground,
                          split_probe, no_local, openephys_path,
                          exdir_path, no_klusta, shutter_channel,
                          no_preprocess, no_spikes, no_lfp, no_tracking):
        settings = config.load_settings()['current']
        if not no_klusta:
            import klusta
            import klustakwik2
        action = None
        if exdir_path is None:
            project = expipe.get_project(PAR.USER_PARAMS['project_id'])
            action = project.require_action(action_id)
            fr = action.require_filerecord()
            if not no_local:
                exdir_path = action_tools._get_local_path(fr)
            else:
                exdir_path = fr.server_path
            exdir_file = exdir.File(exdir_path)
        if openephys_path is None:
            acquisition = exdir_file["acquisition"]
            if acquisition.attrs['acquisition_system'] is None:
                raise ValueError('No Open Ephys aquisition system ' +
                                 'related to this action')
            openephys_session = acquisition.attrs["openephys_session"]
            openephys_path = os.path.join(str(acquisition.directory), openephys_session)
            openephys_base = os.path.join(openephys_path, openephys_session)
            klusta_prm = os.path.abspath(openephys_base) + '.prm'
            prb_path = prb_path or settings.get('probe')
            openephys_file = pyopenephys.File(openephys_path, prb_path)
            print(openephys_path)
            openephys_exp = openephys_file.experiments[0]
            openephys_rec = openephys_exp.recordings[0]
        if not no_preprocess:
            if not pre_filter and not klusta_filter:
                pre_filter = True
            elif pre_filter and klusta_filter:
                raise IOError('Choose either klusta-filter or pre-filter.')
            anas = openephys_rec.analog_signals[0].signal
            fs = openephys_rec.sample_rate.magnitude
            nchan = anas.shape[0]
            sig_tools.create_klusta_prm(openephys_base, prb_path, nchan,
                              fs=fs, klusta_filter=klusta_filter,
                              filter_low=filter_low,
                              filter_high=filter_high,
                              filter_order=filter_order,
                              use_single_threshold=use_single_threshold,
                              threshold_strong_std_factor=threshold_strong_std_factor,
                              threshold_weak_std_factor=threshold_weak_std_factor)
            if pre_filter:
                anas = sig_tools.filter_analog_signals(anas, freq=[filter_low, filter_high],
                                             fs=fs, filter_type='bandpass',
                                             order=filter_order, filter_function=filter_function)
            if len(ground) != 0:
                ground = [int(g) for g in ground]
                anas = sig_tools.ground_bad_channels(anas, ground)
            if split_probe is not None:
                split_chans = np.arange(nchan)
                if split_probe != nchan / 2:
                    warnings.warn('The split probe is not dividing the number' +
                                  ' of channels in two')
                print('Splitting probe in channels \n"' +
                      str(split_chans[:split_probe]) + '"\nand\n"' +
                      str(split_chans[split_probe:]) + '"')
            if common_ref == 'car':
                anas, _ = sig_tools.apply_CAR(anas, car_type='mean',
                                    split_probe=split_probe)
            elif common_ref == 'cmr':
                anas, _ = sig_tools.apply_CAR(anas, car_type='median',
                                    split_probe=split_probe)
            if len(ground) != 0:
                duplicate = [int(g) for g in ground]
                anas = sig_tools.duplicate_bad_channels(anas, duplicate, prb_path)
            sig_tools.save_binary_format(openephys_base, anas)
            if action is not None:
                prepro = {
                    'common_ref': common_ref,
                    'filter': {
                        'pre_filter': pre_filter,
                        'klusta_filter': klusta_filter,
                        'filter_low': filter_low,
                        'filter_high': filter_high,
                    },
                    'grounded_channels': ground,
                    'probe_split': (str(split_chans[:split_probe]) +
                                    str(split_chans[split_probe:]))
                }
                action.require_module(name='preprocessing', contents=prepro,
                                      overwrite=True)

        if not no_klusta:
            print('Running klusta')
            try:
                subprocess.check_output(['klusta', klusta_prm, '--overwrite'])
            except subprocess.CalledProcessError as e:
                raise Exception(e.output)
        if not no_spikes:
            print('Converting from ".kwik" to ".exdir"')
            openephys.generate_spike_trains(exdir_path, openephys_rec,
                                            source='klusta')
            print('Processed spiketrains, manual clustering possible')
        if not no_lfp:
            print('Filtering and downsampling raw data to LFP.')
            openephys.generate_lfp(exdir_path, openephys_rec)
            print('Finished processing LFPs.')
        if not no_tracking:
            print('Converting tracking from OpenEphys raw data to ".exdir"')
            openephys.generate_tracking(exdir_path, openephys_rec)
            # TODO update
            # if shutter_channel is not None:
            #     ttl_times = openephys_file.digital_in_signals[0].times[
            #         shutter_channel]
            #     if len(ttl_times) != 0:
            #         openephys_file.sync_tracking_from_events(ttl_times)
            #     else:
            #         warnings.warn('No TTL events found on IO channel {}'.format(
            #             shutter_channel))


    @cli.command('convert-klusta-oe',
                 short_help='Convert klusta spikes to exdir.')
    @click.argument('action-id', type=click.STRING)
    @click.option('--prb-path',
                  type=click.STRING,
                  help='Path to probefile, assumed to be in expipe config directory by default.',
                  )
    @click.option('--openephys-path',
                  type=click.STRING,
                  help='Path to openeophys dir, if none it is deduced from action id.',
                  )
    @click.option('--exdir-path',
                  type=click.STRING,
                  help='Path to desired exdir directory, if none it is deduced from action id.',
                  )
    @click.option('--no-local',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    @click.option('--nchan',
                  type=click.INT,
                  default=32,
                  help='Number of channels. Default = 32',
                  )
    def generate_klusta_oe(action_id, prb_path, no_local, openephys_path,
                           exdir_path, nchan):
        if openephys_path is None:
            project = expipe.get_project(PAR.USER_PARAMS['project_id'])
            action = project.require_action(action_id)
            fr = action.require_filerecord()
            if not no_local:
                exdir_path = action_tools._get_local_path(fr)
            else:
                exdir_path = fr.server_path
            exdir_file = exdir.File(exdir_path)
            acquisition = exdir_file["acquisition"]
            if acquisition.attrs['acquisition_system'] != 'OpenEphys':
                raise ValueError('No Open Ephys aquisition system ' +
                                 'related to this action')
            openephys_session = acquisition.attrs["openephys_session"]
            openephys_path = os.path.join(str(acquisition.directory), openephys_session)
        prb_path = prb_path or action_tools._get_probe_file('oe', nchan=nchan,
                                               spikesorter='klusta')
        openephys_file = pyopenephys.File(openephys_path, prb_path)
        print('Converting to exdir')
        openephys.generate_spike_trains(exdir_path, openephys_file,
                                            source='klusta')
