# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

import click
from pandas import DataFrame
import yaml

from ifsbench import (cli, DefaultApplication, Benchmark, ScienceSetup, TechSetup,
                      DefaultArch, Job, CpuConfiguration, MpirunLauncher, SrunLauncher, 
                      PydanticConfigMixin, EnvHandler, ConfigMixin)
from ifsbench.data import DataHandler, ExtractHandler, RenameHandler, RenameMode, NamelistHandler, NamelistOverride
from ifsbench.validation import FrameCloseValidation

arches = {
    'default': DefaultArch(
        launcher=MpirunLauncher(),
        cpu_config=CpuConfiguration()
    ),
    'atos':  DefaultArch(
        launcher=SrunLauncher(),
        cpu_config=CpuConfiguration(
            sockets_per_node=2,
            cores_per_socket=64,
            threads_per_core=2,
            gpus_per_node=0
        )
    ),
    'lumi-c':  DefaultArch(
        launcher=SrunLauncher(),
        cpu_config=CpuConfiguration(
            sockets_per_node=2,
            cores_per_socket=64,
            threads_per_core=2,
            gpus_per_node=0
        ),
        account='project_465000527',
        partition='small'
    )
}

def parse_netcdf(path):
    import netCDF4
    import numpy

    result = {}
    rootgrp = netCDF4.Dataset(path, 'r')

    for var_name, value in rootgrp.variables.items():
        n_value = numpy.array(value[:])

        if n_value.ndim != 3:
           continue
       
        mean = n_value.mean(axis=(1,2))
        min = n_value.min(axis=(1,2))
        max = n_value.max(axis=(1,2))

        frame = DataFrame(
            numpy.array([mean, min, max]).T,
            index = [f'Level {l}' for l in range(n_value.shape[0])],
            columns = ['mean', 'min', 'max']
        )

        result[var_name] = frame

    return result

@dataclass
class EclandResult(ConfigMixin):
    frames: Dict[str, DataFrame]
    log: str = None
    walltime: float = None

    @classmethod
    def from_rundir(cls, run_dir):
        frames = {}
        paths = [run_dir/'o_fix.nc', run_dir/'o_gg.nc']

        for path in paths:
            result = parse_netcdf(path)
            frames = {**frames, **result}

        return cls(frames=frames)
    
    def dump_config(
        self, with_class: bool = False
    ) -> Dict[str, Union[str, float, int, bool, List]]:
        config = {
            'log': self.log,
            'walltime': self.walltime,
            'frames': {x: y.to_dict(orient='split') for x,y in self.frames.items()}
        }
        return config
    
    @classmethod
    def from_config(
        cls, config: Dict[str, Union[str, float, int, bool, List, None]]
    ) -> 'PydanticConfigMixin':
        config = dict(config)
        config['frames'] = {x: DataFrame(**y) for x,y in config['frames'].items()}

        return cls(**config)

class EclandScience(PydanticConfigMixin):
    input_archive: Path
    build_dir: Path = None
    namelists: List[NamelistOverride] = None
    env: List[EnvHandler] = None
    tasks: int = 1
    threads: int = 1

class EclandTech(PydanticConfigMixin):
    namelists: List[NamelistOverride] = None
    env: List[EnvHandler] = None
    tasks: int = None

class EclandBenchmark(Benchmark):
    def __init__(self, science, tech):
        eh = ExtractHandler.from_config(config={'archive_path': str(science.input_archive)})

        data_handlers_init = [
            eh,
            RenameHandler(pattern='input$', repl='namelist_template', mode=RenameMode.MOVE)
        ]

        data_handlers_runtime = [
            RenameHandler(pattern='namelist_template', repl='input', mode=RenameMode.COPY)
        ]

        if science.namelists:
            data_handlers_runtime.append(NamelistHandler('namelist_template', 'input', science.namelists))


        env_handlers = []
        
        if science.env:
            env_handlers += science.env

        application = DefaultApplication(
            command = [str(science.build_dir/'bin/ecland-master')],
        )

        science_setup = ScienceSetup(
            data_handlers_init = data_handlers_init,
            data_handlers_runtime = data_handlers_runtime,
            env_handlers = env_handlers,
            application = application
        )


        data_handlers_runtime = []

        if tech.namelists:
            data_handlers_runtime.append(NamelistHandler(
                input_path='namelist_template',
                output_path='input',
                overrides=tech.namelists
            ))

        env_handlers = []
        
        if tech.env:
            env_handlers += science.env

        tech_setup = TechSetup(
            data_handlers_runtime = data_handlers_runtime,
            env_handlers = env_handlers
        )

        super().__init__(science = science_setup, tech=tech_setup)



# Some click-magic is going on here... click will call the callback function
# that is specified in the 'experiment' argument, extract the default run
# options from this experiment file and use them as the default values for
# the argument handling inside the `run_options` wrapper.
@cli.command('from_yaml')
@click.argument('yaml-path', type=click.Path(exists=True))
@click.argument('science', type=str)
@click.option('--build-dir', type=click.Path(exists=True))
@click.option('--tech', type=str, default='default',)
@click.option('--run-dir', type=click.Path(), default=None,
              help='Run directory for the tests (temporary directory by default)')
@click.option('--tasks', type=int, default=None,
              help='Number of tasks to run')
@click.option('--threads', type=int, default=None,
              help='Number of threads to use')
@click.option('--arch', default=None, type=str,
              help='The architecture to use.')
@click.option('--validate', type=click.Path(exists=True),
              help='Validate results against given result file.')
def from_yaml(yaml_path, science, tech, build_dir, run_dir, tasks, threads, arch, validate):
    yaml_path = Path(yaml_path).resolve()

    if run_dir:
        run_dir = Path(run_dir).resolve()

    if build_dir:
        build_dir = Path(build_dir).resolve()

    with yaml_path.open('r') as f:
        yaml_data = yaml.safe_load(f)


    science_data = yaml_data['science'][science]
    tech_data = yaml_data['tech'][tech]

    science_input = EclandScience.from_config(science_data)

    if tech_data:
        tech_input = EclandTech.from_config(tech_data)
    else:
        tech_input = EclandTech()

    benchmark = EclandBenchmark(science = science_input, tech = tech_input)

    if tasks is None:
        tasks = science_input.tasks

    if threads is None:
        threads = science_input.threads

    job = Job(tasks=tasks, cpus_per_task=threads)

    arch = arches.get(arch, arches['default'])

    benchmark.setup_rundir(run_dir)
    benchmark.run(run_dir, job, arch)

    result = EclandResult.from_rundir(run_dir)

    with (run_dir/'result.yaml').open('w') as f:
        yaml.dump(result.dump_config(), f)

    if validate:
        validator = FrameCloseValidation(atol=0, rtol=0)
        with Path(validate).open('r') as f:
            reference = EclandResult.from_config(yaml.safe_load(f))

        if set(result.frames.keys()) != set(reference.frames.keys()):
            raise RuntimeError("Results do not hold the same frames!")
        
        for key in result.frames.keys():
            frame = result.frames[key]
            frame_ref = reference.frames[key]

            equal, mismatch = validator.compare(frame, frame_ref)

            if not equal:
                raise RuntimeError("Results not equal!")


# Some click-magic is going on here... click will call the callback function
# that is specified in the 'experiment' argument, extract the default run
# options from this experiment file and use them as the default values for
# the argument handling inside the `run_options` wrapper.
@cli.command('validate')
@click.argument('result', type=click.Path(exists=True))
@click.argument('reference', type=click.Path(exists=True))
def validate(result, reference):
    validator = FrameCloseValidation(atol=0, rtol=0)

    with Path(result).open('r') as f:
        result = EclandResult.from_config(yaml.safe_load(f))

    with Path(reference).open('r') as f:
        reference = EclandResult.from_config(yaml.safe_load(f))

    if set(result.frames.keys()) != set(reference.frames.keys()):
        raise RuntimeError("Results do not hold the same frames!")
    
    for key in result.frames.keys():
        frame = result.frames[key]
        frame_ref = reference.frames[key]

        equal, mismatch = validator.compare(frame, frame_ref)

        if not equal:
            raise RuntimeError("Results not equal!")

if __name__ == "__main__":
    cli()