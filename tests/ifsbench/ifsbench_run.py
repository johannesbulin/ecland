# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from typing import List

import click
import yaml

from ifsbench import (cli, DefaultApplication, Benchmark, ScienceSetup, TechSetup,
                      DefaultArch, Job, CpuConfiguration, MpirunLauncher, SrunLauncher, 
                      PydanticConfigMixin, EnvHandler)
from ifsbench.data import DataHandler, ExtractHandler, RenameHandler, RenameMode, NamelistHandler, NamelistOverride

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

       result[var_name] = n_value

    return result

class EclandResult:

    def __init__(self, array_dict):
        self._array_dict = dict(array_dict)

    @classmethod
    def from_rundir(cls, run_dir):
        array_dict = {}
        paths = [run_dir/'o_fix.nc', run_dir/'o_gg.nc']

        for path in paths:
            result = parse_netcdf(path)
            array_dict = {**array_dict, **result}

        return cls(array_dict)

    def to_json(self, path):
        import json

        json_result = {x: y.tolist() for x,y in self._array_dict.items()}
        with path.open('w') as f:
            json.dump(json_result, f)

class EclandScience(PydanticConfigMixin):
    input_archive: Path
    build_dir: Path = None
    namelists: List[NamelistOverride] = None
    env: List[EnvHandler] = None
    tasks: int = 1

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
@click.option('--arch', default=None, type=str,
              help='The architecture to use.')
def from_yaml(yaml_path, science, tech, build_dir, run_dir, tasks, arch):
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

    job = Job(tasks=tasks)

    arch = arches.get(arch, arches['default'])

    benchmark.setup_rundir(run_dir)
    benchmark.run(run_dir, job, arch)

    result = EclandResult.from_rundir(run_dir)
    result.to_json(run_dir/'result.json')    

@cli.command('parse')
@click.argument('path', type=click.Path(exists=True))
def parse(path):
    parse_netcdf(path)


if __name__ == "__main__":
    cli()